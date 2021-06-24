import requests

bot_name = "Sam"

if __name__ == "__main__":
    print("Let's data! type 'quit' to exit")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            print("End Chat~~")
            break

        resp = requests.post(url="http://127.0.0.1:5000/", data={"sentence": sentence})
        json_text = resp.json()
        print(f'{bot_name}: {json_text.get("message")}')
