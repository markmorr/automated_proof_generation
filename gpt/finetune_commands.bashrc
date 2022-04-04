set OPENAI_API_KEY = MY_OPEN_AI_KEY
openai api fine_tunes.create -t train-baseline.jsonl -m ada
openai -k "api key" api fine_tunes.create -t trian-baseline.jsonl -m ada


