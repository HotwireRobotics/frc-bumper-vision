from yt_dlp import YoutubeDL

def search_youtube(query="FRC robots 2023", max_results=10):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'skip_download': True,
        'default_search': 'ytsearch',
        'noplaylist': True,
    }

    video_list = []

    with YoutubeDL(ydl_opts) as ydl:
        search = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
        for entry in search['entries']:
            video_list.append({
                'title': entry['title'],
                'url': f"https://www.youtube.com/watch?v={entry['id']}"
            })

    return video_list

# Example usage
if __name__ == "__main__":
    results = search_youtube("FRC robot matches", max_results=5)
    for r in results:
        print(r['title'])
        print(r['url'])
        print("-" * 40)
