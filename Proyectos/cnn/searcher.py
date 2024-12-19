from pytubefix import YouTube
from pytubefix.cli import on_progress
from pytubefix.contrib.search import Search, Filter

SEARCHS = [
    'McLaren Senna exterior design colors',
    
]

for query in SEARCHS:
    results = Search(
        query,
        filters={"duration": Filter.get_duration("4-10 minutes")}
    )
    results.get_next_results()
    print(f"Found {len(results.videos)} videos for query: {query}")
    
    i = 0
    for result in results.videos:
        try:
            yt = YouTube(result.watch_url, on_progress_callback=on_progress)
            video = yt.streams.get_highest_resolution()
            print(f"Downloading video {i} for query: {query}...")
            video.download(output_path=f"src/videos/{query.replace(' ', '_')}", filename=f"{query.replace(' ', '_')}{i}.mp4")
            i += 1
            print(f"Video downloaded successfully for query: {query}")
        except Exception as e:
            print(f"Failed to download video for query: {query}. Error: {e}")