"""Exemplo de uso do módulo cronograma (pode ser rodado direto ou copiado para um notebook)."""
import pandas as pd

from cronograma import catalogo_da_planilha, gerar_cronograma

# 1) Catálogo: vem do módulo de dados (aqui, lido da planilha)
catalogo = catalogo_da_planilha(pd.read_excel("Hortalicas dados reais - FINAL.xlsx"))

# 2) Entradas que virão da interface
demanda = {
    "Cenoura": 2.0, 
    "Alface": 1.0, 
    "Brócolos": 1.0,
    "Tomate": 3.0, 
    "Abobrinha": 1.5, 
    "Pimentão": 0.5, 
    "Couve": 1.0,
    "Cebolinha": 0.2, 
    "Rabanete": 0.5, 
    "Beterraba": 1.5,
    "Batata-doce": 2.0, 
    "Alho": 0.3,
}


# 3) Saídas
def imprimir(resultado):
    print(resultado["status"], "-", resultado["mensagem"])
    for aviso in resultado["avisos"]:
        print("Aviso:", aviso)
    if resultado["status"] in ("otimo", "viavel_tempo_limite"):
        print(pd.Series(resultado["metricas"]).to_string())

        mapa = pd.DataFrame(resultado["mapa"])
        mapa["texto"] = mapa["variante"].fillna("—")
        print("\nMapa do canteiro (linhas = mês, colunas = subárea):")
        print(mapa.pivot(index="mes", columns="subarea", values="texto").to_string())

        print("\nCalendário de tarefas:")
        print(pd.DataFrame(resultado["calendario"]).to_string(index=False))

        print("\nConsumo (kg/mês):")
        consumo = pd.DataFrame(resultado["consumo"])
        print(consumo.pivot(index="produto", columns="mes", values="consumido_kg").to_string())

        print("\nPeríodos vazios e recomendações:")
        for v in resultado["vazios"]:
            print(f"  Subárea {v['subarea']}, meses {v['meses']}: {v['texto']}")

        print("\nHortaliças com demanda que ficaram fora do plano:", resultado["nao_plantadas"])

        print("\nResumo por hortaliça:")
        print(pd.DataFrame(resultado["resumo_hortalicas"]).to_string(index=False))



resultado = gerar_cronograma(
    catalogo, 
    comprimento=2, 
    largura=5, 
    n_subareas=3, 
    demanda=demanda,
    alpha=0.3, 
    beta=0.3, 
    gamma=0.3,
    tempo_limite=480,
)

imprimir(resultado)


