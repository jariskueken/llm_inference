from typing import Any

def gemini_response_to_openai(response: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a Gemini response to an OpenAI response.

    Parameters
    ----------
    response : dict[str, Any]
        The Gemini response to convert.

    Returns
    -------
    dict[str, Any]
        The OpenAI response.

    Sample OpenAI conform response:
    {
        'id': 'chatcmpl-A0y9tqeqV9OXAAIAr4jsk2ufjhEXD',
        'choices':
            [
                {
                    'finish_reason': 'stop',
                    'index': 0,
                    'logprobs': None,
                    'message': {
                        'content': 'Ich kann Ihnen gerne dabei helfen!',
                        'refusal': None,
                        'role': 'assistant',
                        'function_call': None,
                        'tool_calls': None
                        }
                }
            ],
        'created': 1724794613,
        'model': 'gpt-4o-mini-2024-07-18',
        'object': 'chat.completion',
        'service_tier': None,
        'system_fingerprint': 'fp_f33667828e',
        'usage': {
            'completion_tokens': 21,
            'prompt_tokens': 70,
            'total_tokens': 91
            }
    }

    Sample Gemini response:
    {
        'candidates':
            [
                {
                    'content':
                        {
                            'parts':
                                [
                                    {
                                        'text': 'Gerne! Um dir einen passenden
                                    }
                                ],
                            'role': 'model'
                        },
                    'finish_reason': 2,
                    'index': 0,
                    'safety_ratings':
                        [
                            {
                                'category': 9,
                                'probability': 1,
                                'blocked': False
                            },
                            {
                                'category': 8,
                                'probability': 1,
                                blocked': False
                            },
                            {
                                'category': 7,
                                'probability': 1,
                                'blocked': False
                            },
                            {
                                'category': 10,
                                'probability': 1,
                                'blocked': False
                            }
                        ],
                    'token_count': 0,
                    'grounding_attributions': []
                }
            ],
        'usage_metadata':
            {
                'prompt_token_count': 53,
                'candidates_token_count': 100,
                'total_token_count': 153,
                'cached_content_token_count': 0
            }
    }
    """
    openai_conform_response = {}

    if "candidates" in response:
        # candidates should be a list
        assert isinstance(response["candidates"], list), "Candidates should \
be a list."

        # create new key "choices" in openai_conform_response
        openai_conform_response["choices"] = []

        for candidate in response["candidates"]:
            assert isinstance(candidate, dict), "Candidate should be a \
dictionary."

            # create new dictionary for each candidate
            candidate_dict = {}

            if "finish_reason" in candidate:
                candidate_dict["finish_reason"] = candidate["finish_reason"]

            if "index" in candidate:
                candidate_dict["index"] = candidate["index"]

            if "logprops" in candidate:
                candidate_dict["logprops"] = candidate["logprops"]

            # create the message dictionary
            message_dict = {}

            if "content" in candidate:
                assert isinstance(candidate["content"], dict), "Content \
should be a dictionary."

                if "parts" in candidate["content"]:
                    assert isinstance(candidate["content"]["parts"], list), \
                        "Parts should be a list."
                    if len(candidate["content"]["parts"]) > 0:
                        if "text" in candidate["content"]["parts"][0]:
                            message_dict["content"] = \
                                candidate["content"]["parts"][0]["text"]

            # only further refine in message dict is not empty
            if message_dict:
                message_dict["role"] = "assistant"
                message_dict["refusal"] = None
                message_dict["function_call"] = None
                message_dict["tool_calls"] = None

                candidate_dict["message"] = message_dict

            openai_conform_response["choices"].append(candidate_dict)

    metadata_dict = {}
    # finally fix token counts
    if "usage_metadata" in response:
        assert isinstance(response["usage_metadata"], dict), "Usage \
metadata should be a dictionary."

        if "prompt_token_count" in response["usage_metadata"]:
            metadata_dict["prompt_tokens"] = \
                response["usage_metadata"]["prompt_token_count"]

        if "candidates_token_count" in response["usage_metadata"]:
            metadata_dict["completion_tokens"] = \
                response["usage_metadata"]["candidates_token_count"]

        if "total_token_count" in response["usage_metadata"]:
            metadata_dict["total_tokens"] = \
                response["usage_metadata"]["total_token_count"]

    if metadata_dict:
        openai_conform_response["usage"] = metadata_dict

    return openai_conform_response