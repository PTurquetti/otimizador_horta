"""
Módulo de otimização do cronograma de plantio de uma horta doméstica.

Função pública principal:
    gerar_cronograma(catalogo, comprimento, largura, n_subareas, demanda, ...)  ->  dict

Função auxiliar (para o módulo de dados):
    catalogo_da_planilha(df)  ->  list[dict]   (converte a planilha de hortaliças no formato de entrada)

Solver: HiGHS (MIT), usado pelo PuLP 3 (MIT) — gratuitos e sem licença.
Instalação:  python -m pip install "pulp>=3,<4" highspy

Convenções do modelo
- O ano é circular: o plano se repete todo ano (dezembro emenda com janeiro). Meses numerados de 1 a 12.
- Um plantio da variante i no mês j ocupa a subárea de j até j + meses_na_terra - 1 e gera colheita
  em cada mês de j + meses_ate_colheita até j + meses_ate_colheita + duracao_colheita - 1
  (a última colheita acontece no mês em que a terra é liberada).
- A demanda (kg/mês) é a mesma em todos os meses; o excedente colhido não tem valor.
- Objetivo (maximizar):  consumo_total/consumo_maximo  +  alpha * variedade  +  beta * atendimento_do_pior_mes
                         -  gamma * valor_do_excedente/consumo_maximo
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import pulp

MESES = 12
T = list(range(1, MESES + 1))

# Campos que cada linha do catálogo deve ter (formato de entrada do módulo)
CAMPOS_CATALOGO = (
    "id",                  # identificador único da variante
    "nome",                # ex.: "Cenoura de verão"
    "produto",             # ex.: "Cenoura" (opcional: se faltar, é deduzido do nome)
    "familia",             # família botânica (regra de rotação)
    "janela_inicio",       # 1..12: primeiro mês em que pode ser plantada
    "janela_fim",          # 1..12: último mês em que pode ser plantada (pode atravessar o ano)
    "meses_ate_colheita",  # meses do plantio até a 1ª colheita
    "duracao_colheita",    # nº de meses com colheita (1 = colheita única)
    "meses_na_terra",      # meses que a cultura ocupa a subárea
    "prod_mes_kg_m2",      # kg/m² colhidos em CADA mês de colheita
    "preco_kg",            # R$/kg
)

# Recomendações para os períodos vazios (orientação geral; EPAMIG, Circular Técnica 179, 2013)
MESES_FRIOS = {4, 5, 6, 7, 8}   # abril a agosto (Centro-Sul do Brasil)
ADUBOS_VERDES = {
    "frio": ["aveia-preta", "tremoço-branco", "ervilhaca"],
    "calor": ["crotalária", "feijão-de-porco", "mucuna-preta"],
}

_MAPA_PLANILHA = {
    "id": "id",
    "nome": "nome",
    "familia": "familia",
    "inicio janela temporal (mes)": "janela_inicio",
    "fim janela temporal (mes)": "janela_fim",
    "meses até a 1ª colheita": "meses_ate_colheita",
    "duração da colheita (meses)": "duracao_colheita",
    "tempo total na terra (meses)": "meses_na_terra",
    "produtividade por mês de colheita (kg/m²/mês)": "prod_mes_kg_m2",
    "valor comercial varejo (R$/kg)": "preco_kg",
}


# ---------------------------------------------------------------------------------------------
# Função auxiliar de entrada
# ---------------------------------------------------------------------------------------------

def produto_de(nome: str) -> str:
    """'Cenoura de inverno' -> 'Cenoura'; nomes sem variante sazonal ficam iguais."""
    return re.sub(r"\s+de\s+(inverno|ver[aã]o)\s*$", "", str(nome).strip(), flags=re.IGNORECASE)


def catalogo_da_planilha(df) -> list[dict]:
    """Converte a planilha de hortaliças (DataFrame do pandas) no formato de catálogo do módulo."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    faltando = [c for c in _MAPA_PLANILHA if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes na planilha: {faltando}")
    df = df.rename(columns=_MAPA_PLANILHA)
    if "produto" not in df.columns:
        df["produto"] = df["nome"].map(produto_de)
    df["familia"] = df["familia"].fillna("")
    df["prod_mes_kg_m2"] = df["prod_mes_kg_m2"].fillna(0)
    return df[list(CAMPOS_CATALOGO)].to_dict("records")


# ---------------------------------------------------------------------------------------------
# Estruturas internas
# ---------------------------------------------------------------------------------------------

@dataclass
class _Dados:
    var: dict              # id -> dict da variante (campos do catálogo)
    produtos: list         # produtos com demanda > 0 (P_util)
    variantes: dict        # produto -> [ids]
    C: list                # ids das variantes no modelo (C_util)
    familias: list
    janela: dict           # id -> lista de meses de plantio permitidos
    preco_produto: dict    # produto -> R$/kg
    demanda: dict          # produto -> kg/mês
    demanda_mes_rs: float  # R$ de demanda em cada mês
    catalogo_completo: dict  # id -> variante (todas, inclusive sem demanda; usado nas recomendações)
    avisos: list = field(default_factory=list)


@dataclass
class _Modelo:
    prob: pulp.LpProblem
    L: list
    area_sub: float
    x: dict
    z: dict
    zp: dict
    wf: dict
    cons: dict
    s_min: pulp.LpVariable
    consumo_total: object
    div_total: object
    valor_max_consumo: float
    div_max: float


def _meses_janela(inicio: int, fim: int) -> list:
    if inicio == fim:          # convenção: início == fim significa ano inteiro
        return list(T)
    if inicio < fim:
        return list(range(inicio, fim + 1))
    return list(range(inicio, MESES + 1)) + list(range(1, fim + 1))   # atravessa a virada do ano


def _meses_plantio(v: dict, j: int) -> list:
    """Meses de plantio da variante v que geram colheita no mês j (um para cada mês de colheita)."""
    a, d = v["meses_ate_colheita"], v["duracao_colheita"]
    return [(j - k - 1) % MESES + 1 for k in range(a, a + d)]


def _ocupa(j: int, n: int) -> list:
    """Meses ocupados por um plantio no mês j que fica n meses na terra (circular)."""
    return [(j - 1 + k) % MESES + 1 for k in range(n)]


def _num(valor, nome_campo, linha):
    try:
        return float(valor)
    except (TypeError, ValueError):
        raise ValueError(f"Catálogo, '{linha}': campo '{nome_campo}' inválido ({valor!r})")


# ---------------------------------------------------------------------------------------------
# Etapa 1: validação e preparação dos dados
# ---------------------------------------------------------------------------------------------

def _preparar(catalogo, comprimento, largura, n_subareas, demanda, alpha, beta, gamma) -> _Dados:
    if comprimento <= 0 or largura <= 0:
        raise ValueError("Comprimento e largura do canteiro devem ser positivos.")
    if int(n_subareas) != n_subareas or n_subareas < 1:
        raise ValueError("O número de subáreas deve ser um inteiro maior ou igual a 1.")
    if alpha < 0 or beta < 0 or gamma < 0:
        raise ValueError("Os pesos alpha, beta e gamma não podem ser negativos.")
    if not catalogo:
        raise ValueError("O catálogo de hortaliças está vazio.")

    avisos = []
    var = {}
    for linha in catalogo:
        faltando = [c for c in CAMPOS_CATALOGO if c not in linha and c != "produto"]
        if faltando:
            raise ValueError(f"Catálogo, '{linha.get('nome', '?')}': campos ausentes {faltando}")
        v = dict(linha)
        nome = v["nome"]
        v["produto"] = v.get("produto") or produto_de(nome)
        v["familia"] = v.get("familia") or ""
        for c in ("janela_inicio", "janela_fim", "meses_ate_colheita", "duracao_colheita", "meses_na_terra"):
            v[c] = int(_num(v[c], c, nome))
        for c in ("prod_mes_kg_m2", "preco_kg"):
            v[c] = _num(v[c], c, nome)
        if not (1 <= v["janela_inicio"] <= 12 and 1 <= v["janela_fim"] <= 12):
            raise ValueError(f"Catálogo, '{nome}': janela de plantio fora de 1..12")
        if not (1 <= v["meses_na_terra"] <= MESES):
            raise ValueError(f"Catálogo, '{nome}': meses na terra deve estar entre 1 e {MESES}")
        if v["meses_ate_colheita"] < 1 or v["duracao_colheita"] < 1:
            raise ValueError(f"Catálogo, '{nome}': meses até a colheita e duração devem ser >= 1")
        if v["meses_na_terra"] != v["meses_ate_colheita"] + v["duracao_colheita"] - 1:
            avisos.append(f"'{nome}': meses na terra ({v['meses_na_terra']}) diferente de "
                          f"meses até a colheita + duração - 1; o modelo usa os valores como estão.")
        if v["id"] in var:
            raise ValueError(f"Catálogo: id repetido ({v['id']})")
        var[v["id"]] = v

    produtos_catalogo = {v["produto"] for v in var.values()}
    desconhecidos = set(demanda) - produtos_catalogo
    if desconhecidos:
        raise ValueError(f"Hortaliças na demanda que não existem no catálogo: {sorted(desconhecidos)}")
    if any(q < 0 for q in demanda.values()):
        raise ValueError("A demanda não pode ser negativa.")

    produtos = [p for p in dict.fromkeys(v["produto"] for v in var.values()) if demanda.get(p, 0) > 0]
    if not produtos:
        raise ValueError("Nenhuma hortaliça com demanda maior que zero.")

    variantes = {p: [i for i, v in var.items() if v["produto"] == p] for p in produtos}
    C = [i for p in produtos for i in variantes[p]]
    for p in produtos:
        if all(var[i]["prod_mes_kg_m2"] <= 0 for i in variantes[p]):
            avisos.append(f"'{p}': produtividade zero no catálogo; não poderá ser atendida.")

    preco_produto = {p: sum(var[i]["preco_kg"] for i in variantes[p]) / len(variantes[p]) for p in produtos}
    return _Dados(
        var=var,
        produtos=produtos,
        variantes=variantes,
        C=C,
        familias=sorted({var[i]["familia"] for i in C}),
        janela={i: _meses_janela(var[i]["janela_inicio"], var[i]["janela_fim"]) for i in var},
        preco_produto=preco_produto,
        demanda={p: float(demanda[p]) for p in produtos},
        demanda_mes_rs=sum(preco_produto[p] * demanda[p] for p in produtos),
        catalogo_completo=var,
        avisos=avisos,
    )


# ---------------------------------------------------------------------------------------------
# Etapa 2: construção do modelo
# ---------------------------------------------------------------------------------------------

def _construir(d: _Dados, area_total: float, n_subareas: int, alpha: float, beta: float,
               gamma: float) -> _Modelo:
    L = list(range(1, n_subareas + 1))
    A = area_total / n_subareas
    C, P, var = d.C, d.produtos, d.var
    fam_idx = {f: k for k, f in enumerate(d.familias)}   # índices numéricos: nomes de restrição sem acento/espaço
    prod_idx = {p: k for k, p in enumerate(P)}
    prob = pulp.LpProblem("cronograma_horta", pulp.LpMaximize)

    # --- Variáveis
    # x[i, j, l] = 1 se a variante i é plantada no mês j na subárea l (0 fora da janela de plantio)
    x = {(i, j, l): pulp.LpVariable(f"x_{i}_{j}_{l}", 0, 1 if j in d.janela[i] else 0, cat="Binary")
         for i in C for j in T for l in L}
    # z[i, j] = 1 se a variante i é colhida no mês j;  zp[p, j]: idem por hortaliça;  wf[f, j]: por família
    z = {(i, j): pulp.LpVariable(f"z_{i}_{j}", cat="Binary") for i in C for j in T}
    zp = {(p, j): pulp.LpVariable(f"zp_{prod_idx[p]}_{j}", cat="Binary") for p in P for j in T}
    wf = {(f, j): pulp.LpVariable(f"wf_{fam_idx[f]}_{j}", cat="Binary") for f in d.familias for j in T}
    # cons[p, j] = kg da hortaliça p consumidos no mês j;  s_min = fração da demanda (R$) no pior mês
    cons = {(p, j): pulp.LpVariable(f"cons_{prod_idx[p]}_{j}", lowBound=0) for p in P for j in T}
    s_min = pulp.LpVariable("s_min", lowBound=0, upBound=1)

    # --- Colheita e produção por variante
    producao = {}
    for i in C:
        for j in T:
            mps = _meses_plantio(var[i], j)
            plantios = pulp.lpSum(x[i, mp, l] for mp in mps for l in L)
            producao[i, j] = var[i]["prod_mes_kg_m2"] * A * plantios
            prob += z[i, j] <= plantios, f"z_sup_{i}_{j}"
            for mp in mps:
                for l in L:
                    prob += z[i, j] >= x[i, mp, l], f"z_inf_{i}_{j}_{mp}_{l}"

    # --- Colheita e consumo por hortaliça (as variantes se somam)
    for p in P:
        k = prod_idx[p]
        for j in T:
            prob += zp[p, j] <= pulp.lpSum(z[i, j] for i in d.variantes[p]), f"zp_sup_{k}_{j}"
            for i in d.variantes[p]:
                prob += zp[p, j] >= z[i, j], f"zp_inf_{k}_{i}_{j}"
            prob += cons[p, j] <= pulp.lpSum(producao[i, j] for i in d.variantes[p]), f"cons_prod_{k}_{j}"
            prob += cons[p, j] <= d.demanda[p], f"cons_dem_{k}_{j}"

    # --- Famílias colhidas
    for f in d.familias:
        da_familia = [i for i in C if var[i]["familia"] == f]
        for j in T:
            prob += wf[f, j] <= pulp.lpSum(z[i, j] for i in da_familia), f"wf_sup_{fam_idx[f]}_{j}"
            for i in da_familia:
                prob += wf[f, j] >= z[i, j], f"wf_inf_{fam_idx[f]}_{i}_{j}"

    # --- Piso de atendimento mensal (max-min)
    for j in T:
        consumo_mes = pulp.lpSum(d.preco_produto[p] * cons[p, j] for p in P)
        prob += d.demanda_mes_rs * s_min <= consumo_mes, f"piso_{j}"

    # --- Ocupação e rotação (horizonte circular)
    for l in L:
        for j in T:
            # no máximo uma cultura ocupando a subárea em cada mês
            prob += pulp.lpSum(x[i, (j - r - 1) % MESES + 1, l]
                               for i in C for r in range(var[i]["meses_na_terra"])) <= 1, f"ocupa_{j}_{l}"
            # mesma família só volta à subárea após pelo menos 1 mês de intervalo
            for f in d.familias:
                da_familia = [i for i in C if var[i]["familia"] == f]
                prob += pulp.lpSum(x[i, (j - r - 1) % MESES + 1, l]
                                   for i in da_familia
                                   for r in range(min(var[i]["meses_na_terra"] + 1, MESES))) <= 1, \
                    f"familia_{fam_idx[f]}_{j}_{l}"

    # --- Objetivo
    consumo_total = pulp.lpSum(d.preco_produto[p] * cons[p, j] for p in P for j in T)
    div_total = pulp.lpSum(wf.values())
    valor_max = d.demanda_mes_rs * MESES
    div_max = len(d.familias) * MESES
    # valor de mercado de tudo o que é colhido; excedente = colhido - consumido (o que não é consumido se perde)
    colhido_rs = pulp.lpSum(var[i]["preco_kg"] * producao[i, j] for i in C for j in T)
    excedente_rs = colhido_rs - consumo_total
    prob.setObjective(consumo_total * (1 / valor_max)              # consumo total do ano
                      + alpha * (1 / div_max) * div_total          # variedade: famílias colhidas
                      + beta * s_min                               # regularidade: atendimento do pior mês
                      - gamma * (1 / valor_max) * excedente_rs)    # desperdício: valor do excedente

    return _Modelo(prob, L, A, x, z, zp, wf, cons, s_min, consumo_total, div_total, valor_max, div_max)


# ---------------------------------------------------------------------------------------------
# Etapa 3: solver
# ---------------------------------------------------------------------------------------------

def _criar_solver(tempo_limite: float, tolerancia: float):
    """HiGHS via PuLP. Para quando o plano está a até 'tolerancia' do ótimo ou no tempo limite."""
    if not hasattr(pulp, "HiGHS"):
        raise RuntimeError('Esta versão do PuLP não tem a interface do HiGHS. Use:  python -m pip install "pulp>=3,<4"')
    solver = pulp.HiGHS(msg=False, timeLimit=tempo_limite, gapRel=tolerancia)
    if not solver.available():
        raise RuntimeError("HiGHS indisponível. Instale com:  python -m pip install highspy")
    return solver


# ---------------------------------------------------------------------------------------------
# Etapa 4: leitura da solução
# ---------------------------------------------------------------------------------------------

def _v(var) -> float:
    return var.varValue or 0.0


def _vazios(d: _Dados, m: _Modelo, ocupante: dict) -> list:
    """Blocos de meses vazios por subárea, com recomendação de uso."""
    blocos = []
    for l in m.L:
        livres = [j for j in T if ocupante[l, j] is None]
        if not livres:
            continue
        if len(livres) == MESES:
            inicios = [1]
        else:   # início de bloco = mês livre cujo mês anterior (circular) está ocupado
            inicios = [j for j in livres if ocupante[l, (j - 2) % MESES + 1] is not None]
        for s in inicios:
            meses = []
            j = s
            while ocupante[l, j] is None and len(meses) < MESES:
                meses.append(j)
                j = j % MESES + 1
            k = len(meses)
            antes = ocupante[l, (s - 2) % MESES + 1]
            depois = ocupante[l, meses[-1] % MESES + 1]
            fam_vizinhas = {d.var[c]["familia"] for c in (antes, depois) if c is not None}

            # 1) cultura de ciclo curto do catálogo que caiba no bloco (janela e família ok)
            curtas = [v for v in d.catalogo_completo.values()
                      if v["meses_na_terra"] <= k and s in d.janela[v["id"]]
                      and v["familia"] not in fam_vizinhas and v["prod_mes_kg_m2"] > 0]
            curtas.sort(key=lambda v: -v["prod_mes_kg_m2"] * v["duracao_colheita"] * v["preco_kg"])
            if curtas:
                tipo, sugestoes = "ciclo_curto", [v["nome"] for v in curtas[:3]]
                texto = f"Plantar uma cultura de ciclo curto: {', '.join(sugestoes)}."
            elif k >= 2:
                estacao = "frio" if sum(mm in MESES_FRIOS for mm in meses) * 2 >= k else "calor"
                tipo, sugestoes = "adubacao_verde", ADUBOS_VERDES[estacao]
                texto = (f"Adubação verde ({', '.join(sugestoes)}); cortar antes do próximo plantio "
                         f"e deixar a palha sobre o solo.")
            else:
                tipo, sugestoes = "cobertura_morta", []
                texto = "Cobrir o solo com palha ou restos de cultura."
            blocos.append({"subarea": l, "mes_inicio": s, "meses": meses, "duracao": k,
                           "recomendacao": tipo, "sugestoes": sugestoes, "texto": texto})
    return blocos


def _extrair(d: _Dados, m: _Modelo) -> dict:
    var, A = d.var, m.area_sub
    plantios = [(i, j, l) for (i, j, l), xv in m.x.items() if _v(xv) > 0.5]

    # mapa: o que ocupa cada subárea em cada mês e em que fase está
    ocupante = {(l, j): None for l in m.L for j in T}
    fase = {}
    calendario = []
    for i, j, l in plantios:
        v = var[i]
        colheitas = [(j - 1 + k) % MESES + 1
                     for k in range(v["meses_ate_colheita"], v["meses_ate_colheita"] + v["duracao_colheita"])]
        for mm in _ocupa(j, v["meses_na_terra"]):
            ocupante[l, mm] = i
            fase[l, mm] = "plantio" if mm == j else ("colheita" if mm in colheitas else "cultivo")
        calendario.append({"mes": j, "subarea": l, "acao": "plantar", "variante": v["nome"],
                           "produto": v["produto"], "kg": None})
        for mm in colheitas:
            calendario.append({"mes": mm, "subarea": l, "acao": "colher", "variante": v["nome"],
                               "produto": v["produto"], "kg": round(v["prod_mes_kg_m2"] * A, 3)})
    calendario.sort(key=lambda e: (e["mes"], e["acao"] != "colher", e["subarea"]))

    mapa = []
    for l in m.L:
        for j in T:
            i = ocupante[l, j]
            mapa.append({"subarea": l, "mes": j,
                         "variante": var[i]["nome"] if i is not None else None,
                         "produto": var[i]["produto"] if i is not None else None,
                         "fase": fase.get((l, j), "vazio")})

    # consumo por hortaliça e mês
    colhido = {(p, j): 0.0 for p in d.produtos for j in T}
    for e in calendario:
        if e["acao"] == "colher":
            colhido[e["produto"], e["mes"]] += e["kg"]
    consumo = []
    for p in d.produtos:
        for j in T:
            c = max(0.0, _v(m.cons[p, j]))
            consumo.append({"produto": p, "mes": j, "demanda_kg": d.demanda[p],
                            "colhido_kg": round(colhido[p, j], 3), "consumido_kg": round(c, 3),
                            "excedente_kg": round(max(0.0, colhido[p, j] - c), 3)})

    # resumo anual por hortaliça (kg e R$)
    resumo = []
    for p in d.produtos:
        linhas = [c for c in consumo if c["produto"] == p]
        col = sum(c["colhido_kg"] for c in linhas)
        con = sum(c["consumido_kg"] for c in linhas)
        dem = d.demanda[p] * MESES
        resumo.append({"produto": p,
                       "demanda_anual_kg": round(dem, 3),
                       "colhido_kg": round(col, 3),
                       "consumido_kg": round(con, 3),
                       "excedente_kg": round(max(0.0, col - con), 3),
                       "pct_demanda_atendida": round(100 * con / dem, 2),
                       "meses_com_colheita": sum(1 for c in linhas if c["colhido_kg"] > 1e-6),
                       # meses em que toda a demanda do mês foi coberta (tolerância de 1 g para arredondamento)
                       "meses_demanda_atendida": sum(1 for c in linhas if c["consumido_kg"] >= d.demanda[p] - 1e-3),
                       "valor_colhido_rs": round(col * d.preco_produto[p], 2),
                       "valor_consumido_rs": round(con * d.preco_produto[p], 2)})
    colhido_total = sum(r["colhido_kg"] for r in resumo)
    consumido_total = sum(r["consumido_kg"] for r in resumo)
    excedente_total = sum(r["excedente_kg"] for r in resumo)

    atendimento = {j: sum(d.preco_produto[p] * max(0.0, _v(m.cons[p, j])) for p in d.produtos) / d.demanda_mes_rs
                   for j in T}
    pior = min(atendimento, key=atendimento.get)
    consumo_rs = sum(d.preco_produto[p] * max(0.0, _v(m.cons[p, j])) for p in d.produtos for j in T)
    meses_ocupados = sum(1 for k in ocupante.values() if k is not None)
    familias_mes = sum(_v(w) for w in m.wf.values()) / MESES
    hortalicas_mes = sum(_v(w) for w in m.zp.values()) / MESES

    metricas = {
        "objetivo": pulp.value(m.prob.objective),
        "valor_consumido_rs": round(consumo_rs, 2),
        "pct_demanda": round(100 * consumo_rs / m.valor_max_consumo, 2),
        "pct_pior_mes": round(100 * atendimento[pior], 2),
        "mes_pior": pior,
        "familias_por_mes": round(familias_mes, 2),
        "hortalicas_por_mes": round(hortalicas_mes, 2),
        "colhido_total_kg": round(colhido_total, 2),
        "consumido_total_kg": round(consumido_total, 2),
        "excedente_total_kg": round(excedente_total, 2),
        "pct_aproveitamento": round(100 * consumido_total / colhido_total, 1) if colhido_total else 0.0,
        "valor_excedente_rs": round(sum(r["excedente_kg"] * d.preco_produto[r["produto"]] for r in resumo), 2),
        "producao_bruta_rs": round(sum(var[i]["prod_mes_kg_m2"] * var[i]["duracao_colheita"] * var[i]["preco_kg"] * A
                                       for i, _, _ in plantios), 2),
        "pct_area_ocupada": round(100 * meses_ocupados / (len(m.L) * MESES), 1),
        "n_subareas": len(m.L),
        "area_subarea_m2": round(A, 4),
        "n_plantios": len(plantios),
    }
    atendidos = {e["produto"] for e in consumo if e["consumido_kg"] > 1e-6}
    return {
        "metricas": metricas,
        "mapa": mapa,
        "calendario": calendario,
        "consumo": consumo,
        "resumo_hortalicas": resumo,
        "atendimento_mensal": [{"mes": j, "pct": round(100 * atendimento[j], 2)} for j in T],
        "vazios": _vazios(d, m, ocupante),
        "nao_plantadas": [p for p in d.produtos if p not in atendidos],
    }


# ---------------------------------------------------------------------------------------------
# Função pública
# ---------------------------------------------------------------------------------------------

def gerar_cronograma(catalogo: list, comprimento: float, largura: float, n_subareas: int, demanda: dict,
                     alpha: float = 0.2, beta: float = 0.2, gamma: float = 0.0, tempo_limite: float = 60,
                     tolerancia: float = 0.01) -> dict:
    """
    Gera o cronograma anual de plantio que maximiza o atendimento da demanda da família.

    Entradas
        catalogo      lista de dicts com os campos de CAMPOS_CATALOGO (use catalogo_da_planilha para converter)
        comprimento   comprimento do canteiro (m)
        largura       largura do canteiro (m)
        n_subareas    em quantas partes iguais o canteiro é dividido
        demanda       {"Cenoura": 2.0, ...} em kg/mês, igual em todos os meses (0 ou ausente = não quer)
        alpha         peso da variedade (famílias botânicas colhidas por mês); 0 desliga
        beta          peso da regularidade (atendimento do pior mês); 0 desliga
        gamma         peso da penalidade pelo excedente (desperdício); 0 desliga (padrão).
                      Valores a partir de ~0.1 evitam culturas que só podem ser plantadas com muita sobra.
        tempo_limite  segundos máximos de otimização
        tolerancia    distância máxima aceita do ótimo (0.01 = 1%); valores maiores deixam a otimização mais rápida

    Saída: dict com
        status        "otimo" | "viavel_tempo_limite" | "sem_solucao" | "erro_entrada"
        mensagem      explicação curta do status
        avisos        lista de alertas sobre os dados (não impedem a otimização)
        parametros    eco das entradas numéricas
        metricas      indicadores do plano: consumo, colheita total, excedente, aproveitamento, pior mês,
                      variedade, ocupação, ...
        mapa          [{subarea, mes, variante, produto, fase}]  fase: plantio | cultivo | colheita | vazio
        calendario    [{mes, subarea, acao, variante, produto, kg}]  acao: plantar | colher
        consumo       [{produto, mes, demanda_kg, colhido_kg, consumido_kg, excedente_kg}]
        resumo_hortalicas  [{produto, demanda_anual_kg, colhido_kg, consumido_kg, excedente_kg,
                            pct_demanda_atendida, meses_com_colheita, meses_demanda_atendida,
                            valor_colhido_rs, valor_consumido_rs}]
        atendimento_mensal  [{mes, pct}]  % da demanda mensal (R$) atendida
        vazios        [{subarea, mes_inicio, meses, duracao, recomendacao, sugestoes, texto}]
        nao_plantadas hortaliças com demanda que ficaram fora do plano
    """
    t0 = time.time()
    parametros = {"comprimento": comprimento, "largura": largura, "n_subareas": n_subareas,
                  "alpha": alpha, "beta": beta, "gamma": gamma, "tempo_limite": tempo_limite,
                  "tolerancia": tolerancia}
    base = {"status": None, "mensagem": "", "avisos": [], "parametros": parametros, "metricas": {},
            "mapa": [], "calendario": [], "consumo": [], "resumo_hortalicas": [], "atendimento_mensal": [], "vazios": [],
            "nao_plantadas": []}

    try:
        dados = _preparar(catalogo, comprimento, largura, n_subareas, demanda, alpha, beta, gamma)
    except ValueError as e:
        return {**base, "status": "erro_entrada", "mensagem": str(e)}
    base["avisos"] = dados.avisos

    modelo = _construir(dados, comprimento * largura, int(n_subareas), alpha, beta, gamma)
    modelo.prob.solve(_criar_solver(tempo_limite, tolerancia))

    status_lp = pulp.LpStatus.get(modelo.prob.status, "Undefined")
    tem_solucao = any(xv.varValue is not None for xv in modelo.x.values()) and status_lp in ("Optimal", "Not Solved")
    if status_lp == "Optimal":
        sol_status = getattr(modelo.prob, "sol_status", None)
        integer_feasible = getattr(pulp, "LpSolutionIntegerFeasible", 2)
        if sol_status == integer_feasible:
            status, msg = "viavel_tempo_limite", "Parou no tempo limite; o plano é viável, mas pode não ser o melhor."
        else:
            status, msg = "otimo", f"Plano ótimo encontrado (até {100 * tolerancia:g}% do melhor possível)."
    elif tem_solucao:
        status, msg = "viavel_tempo_limite", "Parou no tempo limite; o plano é viável, mas pode não ser o melhor."
    else:
        return {**base, "status": "sem_solucao",
                "mensagem": f"O solver não encontrou solução (status: {status_lp})."}

    resultado = _extrair(dados, modelo)
    resultado["metricas"]["tempo_s"] = round(time.time() - t0, 2)
    resultado["metricas"]["solver"] = "HiGHS"
    return {**base, "status": status, "mensagem": msg, **resultado}
