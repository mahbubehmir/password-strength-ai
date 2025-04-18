# 🔐 تولید و ارزیابی رمز عبور با استفاده از هوش مصنوعی

این پروژه یک نمونه‌ی کاربردی از استفاده از مدل‌های یادگیری عمیق برای مسائل امنیتی در دنیای واقعی است. هدف اصلی، طراحی ابزاری است که بتواند رمزهای عبور تصادفی، اما قابل اطمینان تولید کند و به کمک یک مدل یادگیری ماشین آموزش‌دیده، امنیت آن‌ها را ارزیابی کند.

---

## 📌 مقدمه

رمزهای عبور ضعیف هنوز یکی از دلایل اصلی نشت اطلاعات و حملات سایبری هستند. کاربران معمولاً تمایل دارند رمزهایی انتخاب کنند که به‌خاطر سپردن آن‌ها راحت‌تر باشد، در حالی‌که این موضوع امنیت آن‌ها را کاهش می‌دهد. در این پروژه، سعی شده است با ترکیب روش‌های تولید رمز عبور و استفاده از هوش مصنوعی، ابزاری ایجاد شود که:

- رمز عبورهای قوی‌تری تولید کند.
- سطح امنیت رمزهای عبور را به صورت خودکار ارزیابی نماید.
- یک رابط کاربری ساده برای استفاده کاربران نهایی فراهم کند.

---

## 🧩 ساختار پروژه

### 1. `challesh_4_2.ipynb` (نوت‌بوک آموزشی)

این فایل نقش بخش تحلیلی پروژه را دارد و شامل مراحل زیر است:

- تعریف مسئله و منطق انتخاب روش.
- بررسی نحوه تولید رمز عبور.
- پیاده‌سازی ابتدایی تابع ارزیابی امنیت رمز عبور.
- مقدمه‌ای بر استفاده از مدل‌های زبانی مانند GPT-2 برای طبقه‌بندی امنیت رمزها.
- تحلیل خروجی و تفسیر نتایج.

📌 این فایل مناسب افرادی است که علاقه‌مندند منطق پشت پروژه و مدل‌سازی آن را به‌صورت گام‌به‌گام درک کنند.

---

### 2. `online_4.py` (برنامه Streamlit)

در این فایل، نهایی‌سازی پروژه در قالب یک رابط گرافیکی ساده انجام شده است که شامل موارد زیر است:

- استفاده از مدل GPT-2 آموزش‌دیده برای پیش‌بینی سطح امنیت رمز عبور.
- امکان تنظیم تعداد رمزهای عبور، طول هر رمز و افزودن بخش سفارشی توسط کاربر.
- نمایش نتیجه امنیت برای هر رمز عبور به صورت جداگانه.
- ذخیره‌ی خروجی‌ها در یک فایل متنی جهت استفاده‌های بعدی.

📌 این بخش مناسب کاربران نهایی یا مدیران فناوری اطلاعات است که به ابزاری برای بررسی سریع امنیت رمز عبور نیاز دارند.

---

## 🤖 درباره مدل یادگیری ماشین

### مدل مورد استفاده: `GPT-2 fine-tuned`

- پایه مدل: `GPT-2` از خانواده ترنسفورمرها (Transformer).
- نوع وظیفه: `Sequence Classification` برای طبقه‌بندی رمز عبور به ۳ سطح امنیتی.
- خروجی مدل: یکی از برچسب‌های `["ضعیف", "متوسط", "قوی"]`.

**ویژگی‌های کلیدی مدل:**
- آموزش‌دیده با مجموعه‌ای از رمزهای عبور دسته‌بندی‌شده.
- تشخیص الگوهای ساده، تکراری یا بیش‌ازحد قابل حدس.
- مناسب‌سازی‌شده برای کاربر فارسی‌زبان (نام‌گذاری خروجی‌ها به فارسی).




🎯 کاربردهای واقعی پروژه
پیشنهاد رمز عبور قوی در ثبت‌نام وب‌سایت‌ها یا اپلیکیشن‌ها

تحلیل امنیت رمزهای عبور کاربران توسط تیم‌های امنیت سایبری

ابزار آموزشی در دوره‌های امنیت اطلاعات و هوش مصنوعی

استفاده در UX design برای طراحی سیستم‌های ثبت‌نام امن‌تر


---![Uploading KMPlayer64_KCrvAJuj9D.png…]()


## ⚙️ راه‌اندازی پروژه

### 1. نصب کتابخانه‌های مورد نیاز
```bash
pip install streamlit torch transformers
# password-strength-ai
