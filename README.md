Este é o AffectFace, um programa em Python que utiliza os modelos de 5 emoções ou 8 emoções e o Valence e Arousal do EmoNet.

Para funcionar, você precisa:

1) Instale o git, GCC, MariaDB, libmariadb-dev, cmake, python, pip, opencv e python-opencv:
   sudo apt install git gcc mariadb-server libmariadb-dev cmake python3 python3-pip libopencv-dev python3-opencv

2) clone este repositório usando: git clone https://github.com/welltoncosta/AffectFace
  
3) Baixe o modelo de reconhecimento de landmarks (pontos de controle) e coloque na mesma pasta de affectface.py:  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

4) precisa instalar estes pacotes do python

  pip3 install imutils torch torchvision "opencv-python==4.10.0.84" mariadb --break-system-packages

5) Armazenamento. MariaDB/MySQL

5.1) Crie um usuario com permissão de admin no MariaDB:

sudo mysql

CREATE USER 'ana'@'localhost' IDENTIFIED BY '123';
GRANT ALL PRIVILEGES ON . TO 'ana'@'localhost';
FLUSH PRIVILEGES;

5.2) Crie o banco e a tabela

CREATE DATABASE doutorado;
USE doutorado;

CREATE TABLE face_experimnto (
  id_usuario SERIAL,
  expressao INT,
  valence FLOAT,
  arousal FLOAT,
  horario VARCHAR(100)
);


6) Para executar faça:

6.1) Abra o arquivo affectface.py e ajuste as configurações para seu banco de dados (usuário e senha) que você criou

6.2) certifique-se que a webcam está funcionando (abra o cheese por exemplo e veja se está ok)

6.3) abra o terminal e faça: python3 affectface.py

qualquer dúvida, mande um email para wcoliveira@utfpr.edu.br

att
