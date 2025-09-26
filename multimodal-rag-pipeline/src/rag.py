import json
from langchain_aws import ChatBedrock

def invoke_nova_multimodal(prompt, matched_items):
    system_msg = [
        {"text": "You are a helpful assistant for question answering. "
                 "The text context is relevant information retrieved. "
                 "The provided image(s) are relevant information retrieved."}
    ]

    message_content = []

    for item in matched_items:
        if item['type'] in ['text', 'table']:
            message_content.append({"text": item['text']})
        else:
            message_content.append({"image": {
                "format": "png",
                "source": {"bytes": item['image']},
            }})

    inf_params = {"max_new_tokens": 300, "top_p": 0.9, "top_k": 20}

    message_list = [
        {"role": "user", "content": message_content}
    ]

    message_list.append({"role": "user", "content": [{"text": prompt}]})

    native_request = {
        "messages": message_list,
        "system": system_msg,
        "inferenceConfig": inf_params,
    }

    model_id = "amazon.nova-pro-v1:0"
    client = ChatBedrock(model_id=model_id)

    response = client.invoke(json.dumps(native_request))
    return response.content
