FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN apt update -y && apt install git -y
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
RUN echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
RUN ["/bin/bash", "-c", "source  ~/.bash_profile"]
RUN pyenv install 3.7.0
RUN pip install torch networkx 
