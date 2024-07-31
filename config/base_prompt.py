def get_question_from_prompt(question):
    messages = [
        {"role": "system", "content": "Answer the question without explanation."},
        {"role": "user", "content": 
            f"{question}"
        }
    ]
    return messages

# f"Example:\n\
# Question: Head Entity: Zürich, Relation: /travel/travel_destination/climate./travel/travel_destination_monthly_climate/month Tail Entity: [X], what is X? \n\
# Answer: October.\n\
# Your Task:\n\
# Question: {question} \n\
# Answer: "