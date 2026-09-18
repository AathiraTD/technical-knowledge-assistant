"""Container settings must configure the same client used by ingestion."""
import json
import os
import subprocess
import sys


def test_model_and_cache_environment_are_honoured(tmp_path):
    env = {**os.environ, 'OLLAMA_HOST': 'http://ollama:11434',
           'EMBED_MODEL': 'custom-embedding', 'EMBED_DIMENSIONS': '64',
           'GENERATION_MODEL': 'custom-generation',
           'ASSISTANT_EMBEDDING_CACHE': str(tmp_path / 'cache.db')}
    code = '''import json
from assistant import ollama
from assistant.indexing import embedcache
print(json.dumps([ollama.HOST, ollama.EMBED_MODEL, ollama.EMBED_DIMENSIONS,
                  ollama.GENERATION_MODEL, str(embedcache.DEFAULT_PATH)]))'''
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            check=True, text=True, capture_output=True)
    assert json.loads(result.stdout) == ['http://ollama:11434', 'custom-embedding',
                                       64, 'custom-generation', str(tmp_path / 'cache.db')]
