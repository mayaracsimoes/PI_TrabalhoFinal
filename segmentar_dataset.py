import os
import shutil
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")

def rop_presente(nome_arquivo: str) -> bool:
    tokens = nome_arquivo.split("_")
    diagnostico = int(tokens[5][2:len(tokens[5])])
    return 0 < diagnostico < 9

def segmentar_dataset():
    global DATA_DIR
    # Lista todos os arquivos no diretório
    arquivos = [f for f in os.listdir(DATA_DIR)
                if os.path.isfile(os.path.join(DATA_DIR, f))]

    if not arquivos:
        print("Nenhum arquivo encontrado no diretório.")
        return

    arquivos_processados = 0

    for arquivo in arquivos:
        # Procura pelo padrão no nome do arquivo

        rop = "PRESENTE" if rop_presente(arquivo) else "AUSENTE"

        # Caminhos completos
        caminho_origem = os.path.join(DATA_DIR, arquivo)
        caminho_destino = os.path.join("data\\ROP_" + rop, arquivo)

        # Move o arquivo
        try:
            shutil.move(caminho_origem, caminho_destino)
            arquivos_processados += 1
        except Exception as e:
            print(f"Erro ao mover '{arquivo}': {e}")

    print(f"\nProcessamento concluído! {arquivos_processados} arquivos movidos.")

if __name__ == "__main__":
    segmentar_dataset()