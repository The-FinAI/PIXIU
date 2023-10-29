def process_zhtext(entity_string, text):
    # Initialize
    name = entity_string.split(',')[0]
    if len(entity_string.split(',')) > 1 and entity_string.split(',')[1]:
        entity_type = entity_string.split(',')[1].strip()
    else:
        entity_type = 0
    formatted_name = ' '.join(list(name))
    formatted_result = f"{formatted_name}, {entity_type}"

    entity_list = [(", ".join(val.split(", ")[:-1]), val.split(", ")[-1]) for val in formatted_result.split("\n")]
    text_words = text.split()
    labels = ['O'] * len(text_words)
    text_lower = text

    # Create a list to store the start index of each word
    word_indices = [0]
    for word in text_words[:-1]:
        word_indices.append(word_indices[-1] + len(word) + 1)

    # Iterate over the entity list
    print ("entity_list:",entity_list)
    for entity, entity_type in entity_list:
        entity_words = entity.split()
        entity_lower = entity
        # print ("entity_lower:", entity_lower)

        # Find start and end index of each occurrence of the entity in the text
        start = 0
        while True:
            start = text_lower.find(entity_lower, start)
            if not entity or start == -1: break  # No more occurrence
            end = start + len(entity) - 1

            # Find the words included in this occurrence
            try:
                start_word = next(i for i, ind in enumerate(word_indices) if ind >= start)
                end_word = next(i for i, ind in enumerate(word_indices) if ind > end)

                # Label the words
                labels[start_word] = 'B-' + entity_type
                for i in range(start_word+1, end_word):
                    labels[i] = 'I-' + entity_type

                # Move to the next character after the occurrence
            except Exception:
                pass
            start = end + 1

    return labels
