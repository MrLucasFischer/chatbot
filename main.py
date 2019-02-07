import requests
from flask import Flask, request

session = requests.session()
app = Flask(__name__)



def get_flicker_photo(user_query, server_response):
    """Searchs Flickr for a photo that matches the user query and the retrieved answer for the user query"""

    # TODO pre-process server_response, remove punctuation, stopwords ...
    res = session.get("https://api.flickr.com/services/rest/", params={
        "method": "flickr.photos.search",
        "api_key": "e53b26790d51fd55a2d65d7288dc8ae6",
        "sort": "relevance",
        "text": user_query,
        "tag_mode": "any",
        "tags": ','.join(server_response.split(" ")[:2]),  # TODO change number
        "format": "json",
        "nojsoncallback": "1"
    })

    top_photo_url = "N/A"
    json_response = res.json()  # Get the service response as json
    if json_response["stat"] == "ok":

        photos_arr = json_response["photos"]["photo"]

        # Check if flicker found any photos
        if len(photos_arr) != 0:
            top_photo = photos_arr[0]

            # Assemble the url to the first photo retrieved from flickr
            top_photo_url = "https: //farm{}.staticflickr.com/{}/{}_{}.jpg" \
                .format(top_photo["farm"], top_photo["server"], top_photo["id"], top_photo["secret"])

    else:
        print("An error occurred")

    return top_photo_url


def main():
    while True:
        query = input("Insert you're query :")
        r = session.get("http://localhost:5901/search", params={
            "algo": "bm25",
            "k1": 0.5,
            "b": 0.45,
            "q": query
        })

        get_flicker_photo(query, r.text)


if __name__ == '__main__':
    main()
