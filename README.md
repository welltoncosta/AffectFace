Este é o AffectFace, um programa em Python que utiliza os modelos de 5 emoções ou 8 emoções e o Valence e Arousal do EmoNet.

Para funcionar, você precisa:

1) Baixe o modelo de reconhecimento de landmarks (pontos de controle) e coloque na mesma pasta de affectface.py:  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

2) precisa instalar o GCC, MariaDB, libmariadb-dev, cmake, python, pip, opencv e python-opencv: sudo apt install gcc mariadb-server libmariadb-dev cmake python3 python3-pip libopencv-dev python3-opencv 

3) precisa instalar estes pacotes do python

pip3 install imutils torch torchvision "opencv-python==4.10.0.84" mariadb --break-system-packages

4) Banco de dados e tabela

CREATE DATABASE doutorado;
USE doutorado;

CREATE TABLE face_experimnto (
  id_usuario SERIAL,
  valence FLOAT,
  arousal FLOAT,
  horario VARCHAR(100)
);


5) Para executar faça:

5.1) Ajuste as configurações para seu banco de dados (usuário e senha)

5.2) certifique-se que a webcam está funcionando (abra o cheese por exemplo e veja se está ok)

5.3) abra o terminal e faça ./affectface.py

qualquer dúvida, mande um email para wcoliveira@utfpr.edu.br

att

