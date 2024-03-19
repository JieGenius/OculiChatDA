import os
import time

# os.system("pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 "
#           "--index-url https://download.pytorch.org/whl/cu118")

# issue: https://github.com/InternLM/lmdeploy/issues/1169
os.system('pip install torch==2.0.0 torchvision==0.15.1')
MODEL_DIR = 'models/'
MODEL_REPOSITORY_OPENXLAB = 'flyer/OculiChatDA'
# MODEL_REPOSITORY_OPENXLAB = 'OpenLMLab/internlm2-chat-7b'

if __name__ == '__main__':
    print('服务器cuda版本检测：')
    os.system('ls /usr/local')
    if not os.path.exists(MODEL_DIR):
        os.system(
            'git clone -b v1 --depth 1 https://code.openxlab.org.cn/flyer/OculiChatDA.git {}'
            .format(MODEL_DIR))
        os.system(f'cd {MODEL_DIR} && git lfs pull')
        # download(model_repo=MODEL_REPOSITORY_OPENXLAB, output=MODEL_DIR)

        print('解压后目录结果如下：')
        print(os.listdir(MODEL_DIR))
        os.system("ls -ahl {}".format(MODEL_DIR))
        os.system(f'lmdeploy convert internlm2-chat-7b {MODEL_DIR}')
    import torch
    print('torch.cuda.is_available():', torch.cuda.is_available())
    print('torch.__version__:', torch.__version__)
    print('torch.version.cuda:', torch.version.cuda)
    print('torch.cuda.get_device_name(0):', torch.cuda.get_device_name(0))
    print('nvidia-smi:', os.popen('nvidia-smi').read())

    os.system('lmdeploy serve api_server ./workspace --server-name 0.0.0.0 '
              '--server-port 23333 --tp 1 --cache-max-entry-count 0.5 &')
    time.sleep(5)
    os.system(
        'streamlit run web_demo.py --server.address=0.0.0.0 --server.port 7860'
        ' --server.enableStaticServing True --server.enableStaticServing true')
