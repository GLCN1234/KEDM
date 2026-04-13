"""
netlify/functions/api.py
========================
Netlify serverless function wrapper for the Flask app.
Netlify runs this via python-lambda or netlify-python.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))

from app import app

def handler(event, context):
    """AWS Lambda / Netlify Functions handler."""
    from werkzeug.test import EnvironBuilder
    from werkzeug.wrappers import Request

    method  = event.get('httpMethod', 'GET')
    path    = event.get('path', '/')
    headers = event.get('headers', {}) or {}
    body    = event.get('body', '') or ''
    qs      = event.get('queryStringParameters', {}) or {}

    qs_str = '&'.join(f'{k}={v}' for k, v in qs.items())
    if qs_str:
        path = f'{path}?{qs_str}'

    builder = EnvironBuilder(
        method=method,
        path=path,
        headers=headers,
        data=body,
    )
    env = builder.get_environ()
    with app.request_context(env):
        response = app.full_dispatch_request()
        return {
            'statusCode': response.status_code,
            'headers':    dict(response.headers),
            'body':       response.get_data(as_text=True),
        }
