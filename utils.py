from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from text2digits import text2digits
import torch

# Set device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Initialize text to digit converter
t2d = text2digits.Text2Digits(convert_ordinals=True, add_ordinal_ending=True)

# Load model and tokenizer from local directory
model_path = "./model"

model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def predict(input_text):
    # Encode input transcription to input_ids
    input_ids = tokenizer(input_text,
                          max_length=64,
                          padding='max_length',
                          is_split_into_words=False,
                          return_tensors='pt').input_ids.to(device)

    # Generate prediction
    token_ids = model.generate(input_ids,
                               max_length=64,
                               num_beams=2)

    # Decode prediction into tokens and join into sentence
    output = " ".join(tokenizer.batch_decode(token_ids,
                                             skip_special_tokens=True))
    return output


def remove_disfluency(input_test_data):
    # Convert all the written numbers into digits, if exist
    converted, mapping = t2d.convert(input_test_data)

    # Remove disfluency from transcription, if any
    output = predict(converted)

    # Convert back to written numbers to ensure input/output consistency
    for word, initial in mapping.items():
        if isinstance(initial, str):
            output = output.replace(initial, word)
        else:
            output = output.replace(initial.text(), word)

    return output
