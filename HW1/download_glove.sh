if [ ! -f glove.840B.300d.txt ]; then
  wget https://nlp.stanford.edu/data/glove.840B.300d.zip -O glove.840B.300d.zip
  unzip glove.840B.300d.zip
fi
