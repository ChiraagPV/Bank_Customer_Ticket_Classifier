# Bank Customer Ticket Category Classifier (Fine-Tuned DistilBERT)

## Model Description
This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) for **bank customer complaint classification**.  
It classifies complaints into one of **three categories**:

1. **Credit card or prepaid card**  
2. **Checking or savings account**  
3. **Mortgage**

The model was developed to help banks and financial institutions automatically tag and route complaints to the correct department, improving **efficiency**, **accuracy**, and **response times**.

---

## Intended Uses & Limitations

**Intended Use Cases**
- Automating complaint classification in customer service systems.
- Categorizing historical complaint datasets for analytics.
- Integrating with chatbots or CRM systems for real-time tagging.

**Limitations**
- Only supports English-language inputs.
- Designed specifically for the three categories above — other categories will not be classified correctly.
- May underperform on slang-heavy or incomplete sentences.

---

## Example Inference

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="Chiraag-P-V/bank_customer_ticket_category_classifier_fine_tuned")

text = "I have a credit card issue."
result = classifier(text)

print(result)
# Example output:
# [{'label': 'Credit card or prepaid card', 'score': 0.987}]
