import time
import argparse
import json

import openai


MAX_API_RETRY = 5
REQ_TIME_GAP = 2

def get_eval(user_prompt):
    for i in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=500,
            )
            content = response["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            print(e)
            time.sleep(5)
    print(f"Failed after {MAX_API_RETRY} retries.")
    return "error"


def chatgpt_auto_eval(gt_result, my_test_result):
    print("--------------- Start auto-evaluation, you should verify it does this correctly --------------")
    correct = 0
    user_prompt = f"I am testing whether a LLM model can correctly retreieve the first topic, and would like you to help me judge whether the mode ls correct. Please give me 1 for correct and 0 for incorrect. Only give me a single number. Ignore mistakes if the model is paraphasing or using synonyms. Ignore any simple mistakes such as capitalization and punctuation. The ground truth is {gt_result}, the model prediction is {my_test_result}"

    content = get_eval(user_prompt)

    _correct = content == "1"
    correct += _correct

    output_string = "correct" if _correct else "wrong"

    print(f"Question #{i}: Label: {gt_result}, Predict: {my_test_result} - auto-eval goes with {output_string}")

    # To avoid rate limit by OPENAI
    time.sleep(REQ_TIME_GAP)
    return correct

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str,  default = '/dataheart/ashton/CacheGen2/CacheGen/test_data/16_prompts')
    args = parser.parse_args()
    correct = 0
    
    for i in range(args.start, args.end):
        with open(f"{args.gt_dir}/gt_{i}.txt", "r") as f:
            gt_result = f.read()
        with open(f"{args.results_dir}/{i}_result.txt", "r") as f:
            my_test_result = f.read()
        correct += chatgpt_auto_eval(gt_result, my_test_result)
    
    print(f"accuracy: {correct/ ((args.end - args.start)) }")
