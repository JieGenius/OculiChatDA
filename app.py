import os
import time
import torch
# os.system("pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118")

MODEL_DIR = "models/"
MODEL_REPOSITORY_OPENXLAB = "flyer/OculiChatDA"

if __name__ == '__main__':
    print("服务器cuda版本检测：")
    os.system("ls /usr/local")
    if not os.path.exists(MODEL_DIR):
        from openxlab.model import download

        download(model_repo=MODEL_REPOSITORY_OPENXLAB, output=MODEL_DIR)

        print("解压后目录结果如下：")
        print(os.listdir(MODEL_DIR))
        os.system(f"lmdeploy convert internlm2-chat-7b {MODEL_DIR}")

    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.__version__:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
    print("nvidia-smi:", os.popen('nvidia-smi').read())

    os.system("lmdeploy serve api_server ./workspace --server-name 0.0.0.0 --server-port 23333 --tp 1 --cache-max-entry-count 0.5 &")
    time.sleep(5)
    os.system('streamlit run web_demo.py --server.address=0.0.0.0 --server.port 7860 --server.enableStaticServing True --server.enableStaticServing true')
