import requests

headers = {'Client-ID': 'kimne78kx3ncx6brgo4mv6wki5h1ko'}

# Get app access token
auth_response = requests.post(
    'https://id.twitch.tv/oauth2/token',
    params={'client_id': 'kimne78kx3ncx6brgo4mv6wki5h1ko'},
    data={'client_secret': '', 'grant_type': 'client_credentials'}
)
token = auth_response.json()['access_token']

# Fetch live streams, sort by viewer_count DESC
response = requests.get(
    'https://api.twitch.tv/helix/streams?first=20',
    headers={**headers, 'Authorization': f'Bearer {token}'}
)
streams = sorted(
    response.json()['data'],
    key=lambda x: int(x['viewer_count']),
    reverse=True
)

usernames = [stream['user_name'] for stream in streams]
print(', '.join(f"'{u}'" for u in usernames))