git clone https://github.com/bytedance/SALMONN.git
cd SALMONN

# Download BEATs
wget -O BEATs_iter3.pt 'https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D'

# Download SALMONN checkpoint
wget https://huggingface.co/tsinghua-ee/SALMONN/resolve/main/salmonn_v1.pth?download=true

# Download Whisper v2
git lfs install
git clone https://huggingface.co/openai/whisper-large-v2

# Download Viccuna
git lfs install
git clone https://huggingface.co/lmsys/vicuna-13b-v1.1
