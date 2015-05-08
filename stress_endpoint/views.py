import json
from django.http import StreamingHttpResponse

def main_page(request):
  if request.method == 'POST':
    received_json_data = json.loads(request.body)
    return StreamingHttpResponse('it was post request: ' +
                                 str(received_json_data))
  return StreamingHttpResponse('it was GET request')
