Este é o AffectFace, um programa em Python que utiliza os modelos de 5 emoções ou 8 emoções e o Valence e Arousal do EmoNet.

Para funcionar, você precisa:

1) Instale o git, GCC, MariaDB, libmariadb-dev, cmake, python, pip, opencv e python-opencv:
   sudo apt install git gcc mariadb-server libmariadb-dev cmake python3 python3-pip libopencv-dev python3-opencv

2) clone este repositório usando: git clone https://github.com/welltoncosta/AffectFace
   
3) Baixe esses dois arquivos e cole dentro da pasta emonet:
4) 
Emonet 5 expressões: https://github.com/face-analysis/emonet/raw/refs/heads/master/pretrained/emonet_5.pth

Emonet 8 expressões: https://github.com/face-analysis/emonet/raw/refs/heads/master/pretrained/emonet_8.pth
  
5) Baixe o modelo de reconhecimento de landmarks (pontos de controle) e coloque na mesma pasta de affectface.py:  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

6) precisa instalar estes pacotes do python

  pip3 install imutils torch torchvision "opencv-python==4.10.0.84" mariadb --break-system-packages

6) Armazenamento. MariaDB/MySQL

6.1) Crie um usuario com permissão de admin no MariaDB:

sudo mysql

CREATE USER 'usuario'@'localhost' IDENTIFIED BY '123';

GRANT ALL PRIVILEGES ON . TO 'usuario'@'localhost';

FLUSH PRIVILEGES;

6.2) Crie o banco e a tabela

CREATE DATABASE doutorado;

USE doutorado;

CREATE TABLE face_experimnto (
  id_usuario SERIAL,
  expressao INT,
  valence FLOAT,
  arousal FLOAT,
  horario VARCHAR(100)
);


7) Para executar faça:

8.1) Abra o arquivo affectface.py e ajuste as configurações para seu banco de dados (usuário e senha) que você criou

8.2) certifique-se que a webcam está funcionando (abra o cheese por exemplo e veja se está ok)

8.3) abra o terminal e faça: python3 affectface.py

qualquer dúvida, mande um email para wcoliveira@utfpr.edu.br

att
