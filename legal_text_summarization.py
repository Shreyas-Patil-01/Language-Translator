from transformers import PegasusTokenizer, PegasusForConditionalGeneration

def summarize_legal_text(text):
    # Load Pegasus tokenizer and model
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-legal")
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-legal")

    # Tokenize the input text
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    
    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Example legal text
legal_text = """
This Agreement ("Agreement") is entered into this [Date] by and between [Company Name], a [State] corporation with its principal place of business at [Address] ("Company"), and [Client Name], a [State] corporation with its principal place of business at [Address] ("Client").
1. Services. Company agrees to provide to Client the following services: [Description of services].
2. Compensation. Client agrees to compensate Company for the services rendered pursuant to this Agreement in the amount of [Amount] per [Time period], payable [Payment terms].
3. Term. This Agreement shall commence on [Start Date] and continue in full force and effect until [End Date], unless terminated earlier pursuant to the terms herein.
4. Termination. Either party may terminate this Agreement upon [Notice period] written notice to the other party.
5. Governing Law. This Agreement shall be governed by and construed in accordance with the laws of the State of [State].
"""

# Summarize the legal text
summary = summarize_legal_text(legal_text)
print("Summary:")
print(summary)
