git clone git@hf.co:datasets/ttsds/noise-reference
wget https://huggingface.co/datasets/ttsds/requests/resolve/main/parler_tts_large.tar.gz
mkdir parler_tts_large
tar -xvf parler_tts_large.tar.gz -C parler_tts_large
rm parler_tts_large.tar.gz
wget https://github.com/MiniXC/ttsdb-data/raw/refs/heads/main/tarred_data/speech_libritts_r_test.tar.gz
mkdir speech_libritts_r_test
tar -xvf speech_libritts_r_test.tar.gz -C speech_libritts_r_test
rm speech_libritts_r_test.tar.gz