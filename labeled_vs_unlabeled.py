import numpy as np

def get_weak_labels(texts: list, subreddits: list) -> tuple:
    """
        Assign weak labels based on known NSFW subreddits.
            1 - NSFW subreddits
            0 - general contents
    """
    ### Doing some hard-coding to create a balanced labeled set
    nsfw_subreddits = {
        'nsfw', 'creampies', 'holdthemoan', 'cumsluts', 
        'nude', 'lesbians', 'gaybrosgonewild', 'massivecock', 'DirtySnapchat'
    }

    general_content = {
        'AskReddit', 'technology', 'ask', 'todayilearned'
    }

    labeled_indices = []
    labels_for_labeled = []

    print(f"[DEBUG] Null subreddits: {sum([not isinstance(s, str) for s in subreddits])}")
    ### The above should be 0
    for idx, (text, sub) in enumerate(zip(texts, subreddits)):
        if isinstance(sub, str) and sub.lower() in nsfw_subreddits:
            labeled_indices.append(idx)
            labels_for_labeled.append(1)  
        elif isinstance(sub, str) and sub.lower() in general_content:
            labeled_indices.append(idx)
            labels_for_labeled.append(0)

    return labeled_indices, labels_for_labeled
