def data_formatting(question, chosen, rejected, predict_head):
    if predict_head:
        input_q = f"Complete this triple. Tail entity: {question[1]}, Relation: {question[0]}, Head Entity: "
    else:
        input_q = f"Complete this triple. Head entity: {question[0]}, Relation: {question[1]}, Tail Entity: "
    message = {
        "conversations": [
            {
                "from": "human",
                "value": input_q
            },
        ],
        "chosen": {
            "from": "gpt",
            "value": f"{chosen}"
        },
        "rejected": {
            "from": "gpt",
            "value": f"{rejected}"
        }
    }
    return message