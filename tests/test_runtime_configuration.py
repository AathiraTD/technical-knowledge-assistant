"""Externalised configuration has to reach the module the indexer actually calls.

The Docker requirements ask for configuration to live outside the image, and
the compose file duly sets `OLLAMA_HOST`, `EMBED_MODEL`, `EMBED_DIMENSIONS`,
`GENERATION_MODEL` and `ASSISTANT_EMBEDDING_CACHE`. None of that is worth
anything unless the values land on the objects that ingestion and retrieval
read — `assistant/infrastructure/ollama.py` holds them as module-level
constants, and `assistant/indexing/embedcache.py` resolves `DEFAULT_PATH` the
same way.

Module-level constants are read **once, at import**, which is why this test
spawns a subprocess rather than setting `os.environ` and importing. Patching
the environment in-process after the test session has already imported
`ollama` would assert nothing about a container start, and would pass even if
the modules had been changed to snapshot their defaults at some earlier
moment. A fresh interpreter with the container's environment is the only
faithful reproduction of `docker compose up`.

This is a wiring test and deliberately nothing more. It does not reach Ollama,
does not check that the named models exist, and does not verify that a
different embedding model would be refused at query time — that is the
snapshot compatibility check, tested elsewhere.
"""
import json
import os
import subprocess
import sys


def test_model_and_cache_environment_are_honoured(tmp_path):
    """A fresh interpreter picks up host, both model tags, dimensions and cache path."""
    env = {**os.environ, 'OLLAMA_HOST': 'http://ollama:11434',
           'EMBED_MODEL': 'custom-embedding', 'EMBED_DIMENSIONS': '64',
           'GENERATION_MODEL': 'custom-generation',
           'ASSISTANT_EMBEDDING_CACHE': str(tmp_path / 'cache.db')}
    code = '''import json
from assistant.infrastructure import ollama
from assistant.indexing import embedcache
print(json.dumps([ollama.HOST, ollama.EMBED_MODEL, ollama.EMBED_DIMENSIONS,
                  ollama.GENERATION_MODEL, str(embedcache.DEFAULT_PATH)]))'''
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            check=True, text=True, capture_output=True)
    assert json.loads(result.stdout) == ['http://ollama:11434', 'custom-embedding',
                                       64, 'custom-generation', str(tmp_path / 'cache.db')]
