import random
import string
import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

# بارگذاری مدل آموزش‌دیده‌شده
model_path = "password_classifier"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2ForSequenceClassification.from_pretrained(model_path)
model.eval()  # تنظیم مدل روی حالت ارزیابی (بدون آموزش)

# تابع ارزیابی سطح امنیتی رمز عبور با مدل
def evaluate_password(password):
    inputs = tokenizer(password, return_tensors="pt", padding=True, truncation=True, max_length=32)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    
    # تبدیل برچسب عددی به سطح امنیتی
    labels = ["ضعیف", "متوسط", "قوی"]
    return labels[prediction]

# تابع تولید رمز عبور
def generate_password(length, custom_input=""):
    custom_length = len(custom_input)
    remaining_length = max(length - custom_length, 0)

    chars = string.ascii_letters + string.digits + string.punctuation
    random_part = ''.join(random.choice(chars) for _ in range(remaining_length))
    password = custom_input + random_part
    
    return password

# رابط کاربری در Streamlit
def main():
    st.title("🛡️ تولید و ارزیابی رمز عبور با هوش مصنوعی")

    num_passwords = st.number_input("تعداد رمزهای عبور:", min_value=1, value=5)
    length = st.number_input("طول هر رمز عبور:", min_value=4, value=12)
    custom_input = st.text_input("چه کلمه یا شماره‌هایی می‌خواهید به رمزهای عبور اضافه شود؟ (اختیاری)")

    if st.button("🔑 تولید و ارزیابی رمز عبور"):
        passwords = [generate_password(length, custom_input) for _ in range(num_passwords)]

        st.write("🔍 رمزهای تولید شده و سطح امنیت آن‌ها:")
        for password in passwords:
            strength = evaluate_password(password)
            st.write(f"🔐 {password}  →  **{strength}**")

        with open("passwords_with_strength.txt", "w", encoding="utf-8") as file:
            for password in passwords:
                file.write(f"{password} → {evaluate_password(password)}\n")

        st.success("✅ کلمات عبور همراه با سطح امنیت در فایل passwords_with_strength.txt ذخیره شدند.")

if __name__ == "__main__":
    main()
