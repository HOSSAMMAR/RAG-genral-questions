from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / '.env')
import rag
print('OPENROUTER_API_KEY_PRESENT', bool(rag.OPENROUTER_API_KEY))
print('KEY_PREFIX', rag.OPENROUTER_API_KEY[:10] if rag.OPENROUTER_API_KEY else '')
result = rag.answer_question('What is the capital of France?', number_of_chunks=4)
print('ANSWER', result['answer'])
print('CHUNKS', len(result['retrieved_chunks']))
