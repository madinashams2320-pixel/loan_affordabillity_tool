"""
AI Loan Decision Assistant — Dissertation MVP
Topic: AI-Based Credit Decision Assistant for Responsible Borrowing
in Digital Financial Services (Uzbekistan context)

Deploy:
  - app.py, data_fw.xlsx, requirements.txt in GitHub root
  - Streamlit Cloud → main file: app.py
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
try:
    import joblib as _joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

# ─── Page config — must be FIRST ─────────────────────────────
st.set_page_config(
    page_title="AI Loan Decision Assistant",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Constants ───────────────────────────────────────────────
APP_DIR    = Path(__file__).resolve().parent
DATA_FILE  = APP_DIR / "data_fw.xlsx"
MAIN_SHEET = "Final_Monthly_ML_Dataset"
MC_MODEL_FILE = APP_DIR / "loan_risk_model.pkl"

LANG_OPTIONS = {"Русский": "ru", "English": "en", "O'zbekcha": "uz"}

FEATURE_COLUMNS = [
    "policy_rate_pct","inflation_yoy_pct","cpi_mom_pct",
    "nominal_wage_monthly_approx","real_wage_growth_pct",
    "real_policy_rate_pct","debt_burden_indicator_pct",
]
REQUIRED_COLUMNS = [
    "date","policy_rate_pct","inflation_yoy_pct","cpi_mom_pct",
    "nominal_wage_monthly_approx","real_wage_growth_pct",
    "real_policy_rate_pct","debt_burden_indicator_pct","repayment_pressure_level",
]

# ══════════════════════════════════════════════════════════════
# TRANSLATIONS
# ══════════════════════════════════════════════════════════════
TR: dict[str, dict] = {

# ─── Russian (default) ───────────────────────────────────────
"ru": {
    # steps
    "s1":"Кредит", "s2":"Бюджет", "s3":"Ситуация", "s4":"Результат",
    "btn_back":"← Назад", "btn_next":"Далее →", "btn_calc":"Оценить кредит →",
    "btn_restart":"Начать заново",
    # welcome
    "hero_title":   "💳 AI Loan Decision Assistant",
    "hero_sub":     "Поможем оценить кредит перед тем, как его брать",
    "w_what":       "Что делает этот инструмент",
    "w_body":       "Вы вводите данные о кредите и своём бюджете. Инструмент считает, насколько кредит будет нагрузкой на ваш бюджет каждый месяц — и даёт понятный результат.",
    "w_not":        "Важно понимать:",
    "w_not1":       "Это не одобрение и не отказ в кредите",
    "w_not2":       "Это не финансовый совет",
    "w_not3":       "Это исследовательский инструмент для оценки нагрузки",
    "w_prepare":    "Что нужно знать перед заполнением:",
    "w_p1":         "Сумму кредита и процентную ставку",
    "w_p2":         "Свой ежемесячный доход",
    "w_p3":         "Ежемесячные расходы: обязательные и необязательные",
    "w_p4":         "День получения зарплаты",
    "btn_start":    "Начать оценку →",
    # step 1
    "s1_h":         "Шаг 1 из 3 — Параметры кредита",
    "f_amount":     "Сумма кредита",
    "f_amount_ph":  "Например: 10 000 000",
    "f_rate":       "Годовая процентная ставка (%)",
    "f_rate_ph":    "Например: 24",
    "f_term":       "Срок кредита",
    "f_start":      "Когда планируете взять кредит?",
    "yr":           "лет / г.",
    "term_months":  "месяцев",
    "est_pmt_lbl":  "Примерный ежемесячный платёж",
    "est_total_lbl":"Общая сумма к выплате",
    "v_amount":     "Введите сумму кредита",
    "v_rate":       "Введите процентную ставку",
    # step 2
    "s2_h":         "Шаг 2 из 3 — Ваш бюджет",
    "f_income":     "Ежемесячный доход (после налогов)",
    "f_income_ph":  "Например: 5 000 000",
    "f_ess":        "Обязательные расходы в месяц",
    "f_ess_help":   "Аренда, коммунальные, еда, транспорт — то, без чего нельзя",
    "f_ess_ph":     "Например: 2 000 000",
    "f_flex":       "Необязательные расходы в месяц",
    "f_flex_help":  "Кафе, подписки, шопинг — то, что можно сократить при необходимости",
    "f_flex_ph":    "Например: 500 000",
    "f_exist":      "Текущие выплаты по другим кредитам",
    "f_exist_help": "Если есть — укажите общую сумму в месяц",
    "f_exist_ph":   "Например: 300 000 (или 0)",
    "f_exist_mo":   "Сколько месяцев ещё платить по старым кредитам",
    "f_exist_mo_h": "Через сколько месяцев эти выплаты закончатся? Если нет — поставьте 0",
    "f_sav":        "Финансовый запас / сбережения",
    "f_sav_help":   "Сколько накоплено — на случай непредвиденного",
    "f_sav_ph":     "Например: 1 000 000 (или 0)",
    "f_sal_day":    "В какой день приходит зарплата?",
    "f_pay_day":    "В какой день нужно платить по кредиту?",
    "budget_title": "Как выглядит ваш бюджет",
    "b_income":     "Доход",
    "b_ess":        "Обязательные расходы",
    "b_flex":       "Необязательные расходы",
    "b_exist":      "Выплаты по старым кредитам",
    "b_new":        "Новый платёж по кредиту",
    "b_left":       "Остаток",
    "w_negative":   "⚠️ Судя по данным, этот платёж не вписывается в бюджет. Рассмотрите меньшую сумму или более длинный срок.",
    "w_tight":      "⚠️ После всех платежей остаётся очень мало. Рекомендуем иметь финансовый запас минимум на 1–2 месяца.",
    "v_income":     "Введите ваш доход",
    # step 3
    "s3_h":         "Шаг 3 из 3 — Ваша ситуация",
    "q_stable":     "Ваша зарплата стабильна?",
    "q_stable_y":   "Да, приходит вовремя",
    "q_stable_n":   "Иногда бывают задержки",
    "q_purpose":    "Для чего нужен кредит?",
    "q_purpose_opts":["Срочная необходимость (здоровье, жильё, учёба)",
                      "Бизнес или инвестиции",
                      "Техника, товары, мебель",
                      "Отдых или развлечения",
                      "Погашение другого кредита",
                      "Другое"],
    "q_terms":      "Вы понимаете все условия кредита: комиссии, штрафы, переплату?",
    "q_terms_y":    "Да, разобрался(-ась)",
    "q_terms_n":    "Не до конца",
    "q_delay":      "Если зарплата задержится на 3 дня — сможете ли сделать платёж?",
    "q_delay_y":    "Да, есть запас",
    "q_delay_n":    "Скорее нет",
    "q_refi":       "Этот кредит для погашения другого кредита?",
    "q_refi_y":     "Да",
    "q_refi_n":     "Нет",
    "q_sav_mo":     "На сколько месяцев платежей хватит сбережений?",
    "proj_note":    "Инструмент рассчитывает нагрузку на весь срок кредита, учитывая ожидаемый рост доходов и расходов.",
    "proj_how":     "Как прогнозировать ваш доход?",
    "proj_avg":     "По средней динамике зарплат",
    "proj_own":     "Укажу сам(а)",
    "proj_none":    "Без роста",
    "proj_pct":     "Ожидаемый рост дохода в год (%)",
    "proj_infl":    "Учитывать рост расходов вместе с инфляцией",
    "calc_note":    "Расчёт учитывает ваши данные и базовые экономические параметры (инфляцию, динамику зарплат). Результат — ориентировочный.",
    # result
    "s4_h":         "Результат оценки",
    "r_sub":        "На основе ваших данных",
    "r_lbl_pmt":    "Ежемесячный платёж",
    "r_lbl_pct":    "Доля от дохода",
    "r_lbl_left":   "Остаток после всех платежей",
    "r_lbl_first":  "Первый рискованный месяц",
    "r_na":         "нет",
    # result levels
    "r_low_h":      "✅ Кредит выглядит посильным",
    "r_low_b":      "По вашим данным нагрузка умеренная. Платёж занимает небольшую долю дохода, и после всех обязательных расходов должен оставаться запас.",
    "r_med_h":      "⚠️ Умеренная нагрузка — стоит подготовиться",
    "r_med_b":      "Кредит может быть управляемым, но в некоторые месяцы может быть туго. Рекомендуем проверить детали и иметь финансовый запас.",
    "r_high_h":     "🔶 Высокая нагрузка — рекомендуем пересмотреть условия",
    "r_high_b":     "Кредит будет занимать значительную часть бюджета. Стоит рассмотреть меньшую сумму, более длинный срок или подождать до улучшения финансовой ситуации.",
    "r_crit_h":     "🔴 Кредит может быть финансово рискованным",
    "r_crit_b":     "По вашим данным нагрузка очень высокая. Рекомендуем рассмотреть альтернативы: меньшую сумму, более длинный срок или отложить оформление. Это не финансовый совет — обратитесь к специалисту.",
    # distribution
    "dist_h":       "Как будет меняться нагрузка",
    "dist_low":     "Нормальных месяцев",
    "dist_med":     "Напряжённых месяцев",
    "dist_high":    "Тяжёлых месяцев",
    "dist_crit":    "Критических месяцев",
    # recommendations
    "rec_before_h": "До оформления кредита",
    "rec_reduce_h": "Как снизить нагрузку",
    "rec_timing_h": "Риск по дате платежа",
    "rec_careful_h":"Когда стоит быть особенно осторожным",
    # check items
    "chk_date":     "День платежа раньше дня зарплаты. Попросите банк перенести дату платежа на день после зарплаты — это уберёт ежемесячный риск.",
    "chk_terms":    "Вы не до конца знаете условия. Уточните у банка: общую сумму переплаты, все комиссии, штрафы за досрочное погашение.",
    "chk_total":    "Узнайте полную стоимость кредита — не только ежемесячный платёж, но и итоговую переплату.",
    "chk_buffer":   "Рекомендуем иметь финансовый запас минимум на 1–2 месяца платежей перед подписанием.",
    "chk_delay":    "Вы отметили, что при задержке зарплаты сделать платёж будет сложно. Создайте небольшой резерв — хотя бы один платёж.",
    # reduce
    "red_amount":   "Уменьшить сумму кредита — это снизит ежемесячный платёж.",
    "red_term":     "Увеличить срок — меньше платить каждый месяц (но больше в итоге).",
    "red_flex":     "Временно сократить необязательные расходы — кафе, подписки, шопинг.",
    "red_no_extra": "Не брать новые кредиты, пока этот не выплачен.",
    # timing
    "tim_date":     "День платежа ({p}) раньше дня зарплаты ({s}). Это создаёт ежемесячный риск — деньги могут ещё не поступить.",
    "tim_ok":       "День платежа ({p}) после зарплаты ({s}) — это хорошо. Риск по дате платежа минимальный.",
    # careful
    "car_salary":   "Зарплата может задерживаться. Держите запас минимум на 2 платежа на отдельном счёте.",
    "car_non_urg":  "Кредит берётся не на срочные нужды. При высокой нагрузке лучше подождать или взять меньше.",
    "car_refi":     "Новый кредит для погашения старого — высокий риск долговой ловушки. Сначала обсудите реструктуризацию с банком.",
    "car_low_inc":  "Ваш доход ниже среднего по стране. Платёж будет занимать значительную долю бюджета.",
    "car_macro":    "Экономические условия создают дополнительное давление на бюджет. Рекомендуем взять меньше или подождать.",
    "car_flex":     "Большая часть необязательных расходов. Если придётся их сократить, это освободит часть бюджета.",
    # expanders
    "exp_charts":   "Подробный прогноз по месяцам",
    "exp_table":    "Таблица симуляции",
    "exp_macro":    "Какие экономические параметры используются",
    "exp_method":   "Как считается результат",
    "ch1_title":    "Сколько дохода уходит на кредит — по месяцам",
    "ch2_title":    "Остаток после платежей — по месяцам",
    "ml_proba_title": "Уверенность модели (Random Forest) — вероятность уровня нагрузки по месяцам",
    "macro_note":   "Эти данные используются только внутри расчёта. Вам не нужно их анализировать — они влияют на прогноз нагрузки.",
    "download":     "Скачать таблицу (CSV)",
    "method_body":  """**Ежемесячный платёж** рассчитан по формуле аннуитета (равные платежи каждый месяц).

**Макроэкономический уровень нагрузки** определяется обученной моделью **Random Forest** (200 деревьев, class_weight=balanced). Модель обучена на исторических месячных данных Узбекистана (2010–2025, 192 наблюдения). Признаки: инфляция, ставка рефинансирования, CPI MoM, номинальная зарплата, реальный рост зарплат, реальная ставка, индикатор долговой нагрузки. Целевая переменная: Low / Medium / High.

По результатам анализа (диссертация): точность Random Forest на тестовой выборке — 92.3%, F1-macro — 0.651. Decision Tree: точность 79.5%, F1-macro 0.760. Логистическая регрессия показала наибольшую стабильность при временно́й кросс-валидации (mean F1 0.58).

**Нагрузка по месяцам** дополнительно оценивается исходя из:
- доли платежа от дохода (каждый месяц)
- остатка после всех расходов и платежей
- наличия финансового запаса
- даты платежа и даты зарплаты

**Уровни нагрузки (финальные):**
- ✅ Нормальная — платёж занимает до 25% дохода, запас есть
- ⚠️ Умеренная — 25–40%, иногда может быть напряжённо
- 🔶 Высокая — 40–50%, рекомендуем пересмотреть условия
- 🔴 Критическая — больше 50%, высокий финансовый риск

Инструмент не принимает решений за банк и не является финансовым советником.""",
    # pressure labels
    "lbl_low":"Нормальная","lbl_med":"Умеренная","lbl_high":"Высокая","lbl_crit":"Критическая",
    # table columns
    "tbl_date":"Дата","tbl_month":"Месяц","tbl_pmt":"Платёж","tbl_inc":"Доход",
    "tbl_ess":"Обяз.","tbl_flex":"Необяз.","tbl_pti":"PTI %","tbl_tdb":"TDB %",
    "tbl_cash":"Остаток","tbl_bal":"Баланс","tbl_lvl":"Уровень",
    # custom term
    "term_custom":"Свой срок (месяцев)","term_mode_pre":"Выбрать из списка","term_mode_own":"Ввести вручную",
    # validation
    "v_rate_high":"Ставка выше 100% — проверьте, нет ли ошибки.",
    "v_exp_high":"Сумма расходов превышает доход — проверьте введённые данные.",
    "v_savings_low":"Сбережений меньше одного платежа. Рекомендуем создать запас перед оформлением.",
    # AI features
    # Monte Carlo ML model
    "mc_default_lbl":   "Вероятность дефолта (ML)",
    "mc_default_sub":   "Monte Carlo · Random Forest",
    "mc_no_model":      "Файл loan_risk_model.pkl не найден. Запустите model_training.py локально.",
    # Smart insights
    "si_title":   "🧠 Персональный анализ",
    "si_empty":   "Нет замечаний — ситуация выглядит нормально.",
    "si_critical":"Критично","si_warning":"Внимание","si_tip":"Совет","si_ok":"Хорошо",
    # errors
    "err_dataset":  "Файл data_fw.xlsx не найден. Поместите его в ту же папку GitHub, что и app.py.",
    "err_forecast": "Не удалось построить прогноз на весь срок. Проверьте data_fw.xlsx.",
    "disclaimer":   "Исследовательский инструмент. Не является финансовым советом, кредитным скорингом или решением банка. Не одобряет и не отказывает в кредите.",
},

# ─── English ─────────────────────────────────────────────────
"en": {
    "s1":"Loan","s2":"Budget","s3":"Situation","s4":"Result",
    "btn_back":"← Back","btn_next":"Next →","btn_calc":"Estimate loan →",
    "btn_restart":"Start over",
    "hero_title":"💳 AI Loan Decision Assistant",
    "hero_sub":  "Estimate whether a loan will be affordable — before you apply",
    "w_what":    "What this tool does",
    "w_body":    "You enter your loan details and budget. The tool calculates how much pressure the loan puts on your monthly budget — and gives you a clear result.",
    "w_not":     "Important to understand:",
    "w_not1":    "This is not a loan approval or rejection",
    "w_not2":    "This is not financial advice",
    "w_not3":    "This is a research tool for estimating loan affordability",
    "w_prepare": "What you will need:",
    "w_p1":      "Loan amount and interest rate",
    "w_p2":      "Your monthly net income",
    "w_p3":      "Your essential and optional monthly expenses",
    "w_p4":      "Your salary day and expected payment due day",
    "btn_start": "Start estimation →",
    "s1_h":      "Step 1 of 3 — Loan details",
    "f_amount":  "Loan amount","f_amount_ph":"e.g. 10 000 000",
    "f_rate":    "Annual interest rate (%)","f_rate_ph":"e.g. 24",
    "f_term":    "Loan term","f_start":"When do you plan to take the loan?",
    "yr":        "year(s)","term_months":"months",
    "est_pmt_lbl":"Estimated monthly payment","est_total_lbl":"Total repayable amount",
    "v_amount":"Please enter the loan amount","v_rate":"Please enter the interest rate",
    "s2_h":      "Step 2 of 3 — Your budget",
    "f_income":  "Monthly net income (after taxes)","f_income_ph":"e.g. 5 000 000",
    "f_ess":     "Essential monthly expenses","f_ess_help":"Rent, utilities, food, transport — unavoidable costs",
    "f_ess_ph":  "e.g. 2 000 000",
    "f_flex":    "Optional monthly expenses","f_flex_help":"Dining, subscriptions, shopping — can be reduced if needed",
    "f_flex_ph": "e.g. 500 000",
    "f_exist":   "Existing loan payments / month","f_exist_help":"Current total monthly loan instalments",
    "f_exist_ph":"e.g. 300 000 (or 0)",
    "f_exist_mo":"Remaining months on existing loans",
    "f_exist_mo_h":"After this many months, existing loan payments stop. Enter 0 if none.",
    "f_sav":     "Savings / emergency buffer","f_sav_help":"Your current savings amount",
    "f_sav_ph":  "e.g. 1 000 000 (or 0)",
    "f_sal_day": "Salary day of month","f_pay_day":"Loan payment due day",
    "budget_title":"Your monthly budget preview",
    "b_income":"Income","b_ess":"Essential expenses","b_flex":"Optional expenses",
    "b_exist":"Existing loan payments","b_new":"New loan payment","b_left":"Money left",
    "w_negative":"⚠️ The new payment does not fit the current budget. Consider a smaller amount or longer term.",
    "w_tight":   "⚠️ Very little money left after all payments. Try to have at least 1–2 months of payments in savings.",
    "v_income":  "Please enter your income",
    "s3_h":      "Step 3 of 3 — Your situation",
    "q_stable":  "Is your salary stable?","q_stable_y":"Yes, arrives on time","q_stable_n":"Sometimes delayed",
    "q_purpose": "What is this loan for?",
    "q_purpose_opts":["Urgent necessity (health, housing, education)","Business or investment",
                      "Electronics, goods, furniture","Leisure or travel",
                      "Paying off another loan","Other"],
    "q_terms":   "Do you understand all loan conditions — fees, penalties, total cost?",
    "q_terms_y": "Yes, fully","q_terms_n":"Not fully",
    "q_delay":   "If salary is delayed 3 days, can you still make the payment?",
    "q_delay_y": "Yes, I have a buffer","q_delay_n":"Probably not",
    "q_refi":    "Is this loan to pay off another existing loan?","q_refi_y":"Yes","q_refi_n":"No",
    "q_sav_mo":  "How many months of loan payments can your savings cover?",
    "proj_note": "The tool simulates loan pressure over the full loan term, factoring in expected income and expense changes.",
    "proj_how":  "How to project your income?","proj_avg":"Follow average salary trends",
    "proj_own":  "Enter my own estimate","proj_none":"Assume no growth",
    "proj_pct":  "Expected annual income growth (%)","proj_infl":"Let expenses grow with expected price growth",
    "calc_note": "The calculation uses your budget data and basic economic assumptions (inflation, salary trends). The result is approximate.",
    "s4_h":      "Your loan assessment","r_sub":"Based on your inputs",
    "r_lbl_pmt": "Monthly payment","r_lbl_pct":"Share of income",
    "r_lbl_left":"Money left after all payments","r_lbl_first":"First risky month","r_na":"none",
    "r_low_h":   "✅ Loan looks affordable",
    "r_low_b":   "Based on your data, the loan pressure is manageable. The payment is a reasonable share of income and there should be money left after all expenses.",
    "r_med_h":   "⚠️ Moderate pressure — prepare carefully",
    "r_med_b":   "The loan may be manageable overall, but some months could be tight. We recommend checking the details and keeping a financial buffer.",
    "r_high_h":  "🔶 High pressure — consider revising the terms",
    "r_high_b":  "The loan will take up a significant portion of your budget. Consider a smaller amount, longer term, or waiting until your financial situation improves.",
    "r_crit_h":  "🔴 Loan may be financially risky",
    "r_crit_b":  "Based on your data, the loan pressure is very high. We recommend exploring alternatives: smaller amount, longer term, or postponing. This is not financial advice.",
    "dist_h":"Pressure distribution over loan term","dist_low":"Low-pressure months",
    "dist_med":"Moderate months","dist_high":"High-pressure months","dist_crit":"Critical months",
    "rec_before_h":"Before applying","rec_reduce_h":"How to reduce pressure",
    "rec_timing_h":"Payment timing","rec_careful_h":"When to be especially careful",
    "chk_date":  "Payment day is before salary day. Ask the bank to move the payment date to after your salary arrives.",
    "chk_terms": "You don't fully know the conditions. Ask the bank: total cost, all fees, penalties and early repayment rules.",
    "chk_total": "Find out the full loan cost — not just monthly payment, but total repayable amount.",
    "chk_buffer":"Have at least 1–2 months of payments saved up before signing.",
    "chk_delay": "You indicated you may not be able to pay if salary is delayed. Build a small reserve — at least one payment.",
    "red_amount":"Reduce the loan amount to lower the monthly payment.",
    "red_term":  "Choose a longer term — lower monthly payment (but higher total cost).",
    "red_flex":  "Temporarily cut optional spending — dining, subscriptions, shopping.",
    "red_no_extra":"Avoid taking new loans while this one is active.",
    "tim_date":  "Payment day ({p}) is before salary day ({s}). This creates a monthly timing risk.",
    "tim_ok":    "Payment day ({p}) is after salary day ({s}) — good. Timing risk is minimal.",
    "car_salary":"Salary may be delayed. Keep at least 2 payments in a separate accessible account.",
    "car_non_urg":"Loan is for non-essential purpose. If pressure is high, consider waiting or reducing the amount.",
    "car_refi":  "New loan to pay off old loan — high risk of a debt cycle. First discuss restructuring with the bank.",
    "car_low_inc":"Your income is below the national average. The payment will take a high share of your budget.",
    "car_macro": "Economic conditions are adding pressure. Consider borrowing less or waiting.",
    "car_flex":  "Large optional expenses. Reducing them could free up budget if needed.",
    "exp_charts":"Detailed monthly forecast","exp_table":"Monthly simulation table",
    "exp_macro": "Economic assumptions used in the calculation","exp_method":"How the result is calculated",
    "ch1_title": "Loan payment as % of income — month by month",
    "ch2_title": "Money left after payments — month by month",
    "ml_proba_title": "Model confidence (Random Forest) — predicted pressure probability per month",
    "macro_note":"These parameters are used internally. You do not need to analyse them — they affect the pressure forecast.",
    "download":  "Download table (CSV)",
    "method_body":"""**Monthly payment** is calculated using the annuity formula (equal monthly payments).

**Macro-level pressure** is classified by a trained **Random Forest** model (200 trees, class_weight=balanced), fit on monthly Uzbekistan macro-financial data (2010–2025, 192 observations). Features: inflation, policy rate, CPI MoM, nominal wage, real wage growth, real policy rate, debt burden indicator. Target: Low / Medium / High.

Model performance (dissertation analysis): Random Forest test accuracy 92.3%, F1-macro 0.651. Decision Tree: accuracy 79.5%, F1-macro 0.760. Logistic Regression showed the most stable time-series cross-validation performance (mean F1 0.58).

**Monthly pressure** is further scored based on:
- Payment as % of income each month
- Money left after all expenses and payments
- Savings buffer
- Payment timing vs salary date

**Final pressure levels:**
- ✅ Low — payment is up to 25% of income, buffer exists
- ⚠️ Moderate — 25–40%, some months may be tight
- 🔶 High — 40–50%, consider revising terms
- 🔴 Critical — above 50%, significant financial risk

The tool does not make bank decisions or provide financial advice.""",
    "lbl_low":"Low","lbl_med":"Moderate","lbl_high":"High","lbl_crit":"Critical",
    # table columns
    "tbl_date":"Date","tbl_month":"Month","tbl_pmt":"Payment","tbl_inc":"Income",
    "tbl_ess":"Essential","tbl_flex":"Optional","tbl_pti":"PTI %","tbl_tdb":"TDB %",
    "tbl_cash":"Cash left","tbl_bal":"Balance","tbl_lvl":"Level",
    # custom term
    "term_custom":"Custom term (months)","term_mode_pre":"Choose from list","term_mode_own":"Enter manually",
    # validation
    "v_rate_high":"Interest rate above 100% — please double-check.",
    "v_exp_high":"Total expenses exceed income — please verify your inputs.",
    "v_savings_low":"Savings are less than one monthly payment. We recommend building a buffer before applying.",
    # AI features
    # Monte Carlo ML model
    "mc_default_lbl":   "Default probability (ML)",
    "mc_default_sub":   "Monte Carlo · Random Forest",
    "mc_no_model":      "loan_risk_model.pkl not found. Run model_training.py locally first.",
    # Smart insights
    "si_title":   "🧠 Personal analysis",
    "si_empty":   "No issues found — situation looks manageable.",
    "si_critical":"Critical","si_warning":"Warning","si_tip":"Tip","si_ok":"Good",
    # errors
    "err_dataset":"data_fw.xlsx not found. Place it in the same GitHub folder as app.py.",
    "err_forecast":"Could not build forecast for the full loan term. Check data_fw.xlsx.",
    "disclaimer":"Research tool. Not financial advice, not a credit score, not a bank decision. Does not approve or reject loans.",
},

# ─── Uzbek Latin ─────────────────────────────────────────────
"uz": {
    "s1":"Kredit","s2":"Byudjet","s3":"Vaziyat","s4":"Natija",
    "btn_back":"← Orqaga","btn_next":"Keyingi →","btn_calc":"Baholash →",
    "btn_restart":"Qaytadan boshlash",
    "hero_title":"💳 AI Loan Decision Assistant",
    "hero_sub":  "Kreditni olishdan oldin uning byudjetga ta'sirini baholang",
    "w_what":    "Bu vosita nima qiladi",
    "w_body":    "Kredit va byudjet ma'lumotlarini kiritasiz. Vosita har oyda kreditning byudjetingizga qancha yuklama qilishini hisoblaydi.",
    "w_not":     "Muhim:",
    "w_not1":    "Bu kredit tasdiqlash yoki rad etish emas",
    "w_not2":    "Bu moliyaviy maslahat emas",
    "w_not3":    "Bu kredit yuklamasini baholash uchun tadqiqot vositasi",
    "w_prepare": "Boshlashdan oldin kerak bo'ladiganlar:",
    "w_p1":      "Kredit miqdori va foiz stavkasi",
    "w_p2":      "Oylik sof daromad",
    "w_p3":      "Majburiy va ixtiyoriy oylik xarajatlar",
    "w_p4":      "Ish haqi kuni va to'lov muddati kuni",
    "btn_start": "Baholashni boshlash →",
    "s1_h":      "1-qadam / 3 — Kredit parametrlari",
    "f_amount":  "Kredit miqdori","f_amount_ph":"Masalan: 10 000 000",
    "f_rate":    "Yillik foiz stavkasi (%)","f_rate_ph":"Masalan: 24",
    "f_term":    "Kredit muddati","f_start":"Kreditni qachon olishni rejalashtirmoqdasiz?",
    "yr":        "yil","term_months":"oy",
    "est_pmt_lbl":"Taxminiy oylik to'lov","est_total_lbl":"Jami to'lanadigan summa",
    "v_amount":"Kredit miqdorini kiriting","v_rate":"Foiz stavkasini kiriting",
    "s2_h":      "2-qadam / 3 — Byudjetingiz",
    "f_income":  "Oylik sof daromad","f_income_ph":"Masalan: 5 000 000",
    "f_ess":     "Majburiy oylik xarajatlar","f_ess_help":"Ijara, kommunal, oziq-ovqat, transport",
    "f_ess_ph":  "Masalan: 2 000 000",
    "f_flex":    "Ixtiyoriy oylik xarajatlar","f_flex_help":"Kafe, obunalar, xarid — kerak bo'lsa kamaytiriladi",
    "f_flex_ph": "Masalan: 500 000",
    "f_exist":   "Mavjud kredit to'lovlari / oy","f_exist_help":"Hozir to'layotgan kredit to'lovlari jami",
    "f_exist_ph":"Masalan: 300 000 (yoki 0)",
    "f_exist_mo":"Mavjud kreditlarning qolgan muddati (oy)",
    "f_exist_mo_h":"Shu oylardan so'ng mavjud to'lovlar to'xtaydi. Yo'q bo'lsa 0 kiriting.",
    "f_sav":     "Jamg'arma / moliyaviy bufer","f_sav_help":"Hozirgi jamg'arma miqdori",
    "f_sav_ph":  "Masalan: 1 000 000 (yoki 0)",
    "f_sal_day": "Ish haqi keladigan kun","f_pay_day":"Kredit to'lovi muddati kuni",
    "budget_title":"Oylik byudjetingiz",
    "b_income":"Daromad","b_ess":"Majburiy xarajatlar","b_flex":"Ixtiyoriy xarajatlar",
    "b_exist":"Mavjud kredit to'lovlari","b_new":"Yangi kredit to'lovi","b_left":"Qoldiq",
    "w_negative":"⚠️ Yangi to'lov joriy byudjetga sig'maydi. Miqdorni kamaytiring yoki muddatni uzaytiring.",
    "w_tight":   "⚠️ Barcha to'lovlardan so'ng juda kam qoladi. Kamida 1–2 oylik to'lov uchun jamg'arma bo'lsin.",
    "v_income":  "Daromadingizni kiriting",
    "s3_h":      "3-qadam / 3 — Vaziyatingiz",
    "q_stable":  "Ish haqingiz barqarormi?","q_stable_y":"Ha, o'z vaqtida keladi","q_stable_n":"Ba'zan kechikadi",
    "q_purpose": "Kredit nima uchun kerak?",
    "q_purpose_opts":["Shoshilinch ehtiyoj (sog'liq, uy-joy, ta'lim)","Biznes yoki investitsiya",
                      "Texnika, tovar, mebel","Dam olish yoki sayohat",
                      "Boshqa kreditni to'lash","Boshqa"],
    "q_terms":   "Kredit shartlarini to'liq bilasizmi: komissiyalar, jarimalar, jami narxi?",
    "q_terms_y": "Ha, to'liq","q_terms_n":"To'liq emas",
    "q_delay":   "Ish haqi 3 kun kechiksa, to'lovni amalga oshira olasizmi?",
    "q_delay_y": "Ha, zahiram bor","q_delay_n":"Qiyin bo'ladi",
    "q_refi":    "Bu kredit boshqa kreditni to'lash uchunmi?","q_refi_y":"Ha","q_refi_n":"Yo'q",
    "q_sav_mo":  "Jamg'arma necha oylik kredit to'lovini qoplaydi?",
    "proj_note": "Vosita butun kredit muddati bo'yicha yuklamani hisoblaydi.",
    "proj_how":  "Daromadni qanday prognoz qilish?","proj_avg":"O'rtacha ish haqi dinamikasiga ko'ra",
    "proj_own":  "O'zim kiritaman","proj_none":"O'smaydi deb hisoblash",
    "proj_pct":  "Kutilayotgan yillik daromad o'sishi (%)","proj_infl":"Xarajatlar inflyatsiya bilan o'ssin",
    "calc_note": "Hisob-kitob byudjet ma'lumotlaringiz va asosiy iqtisodiy parametrlarni hisobga oladi. Natija taxminiy.",
    "s4_h":      "Baholash natijasi","r_sub":"Ma'lumotlaringiz asosida",
    "r_lbl_pmt": "Oylik to'lov","r_lbl_pct":"Daromadning ulushi",
    "r_lbl_left":"Barcha to'lovlardan keyingi qoldiq","r_lbl_first":"Birinchi xavfli oy","r_na":"yo'q",
    "r_low_h":   "✅ Kredit ko'tarilishi mumkin ko'rinadi",
    "r_low_b":   "Ma'lumotlaringizga ko'ra yuklama o'rtacha. To'lov daromadning kichik qismini egallaydi.",
    "r_med_h":   "⚠️ O'rtacha yuklama — tayyorgarlik ko'ring",
    "r_med_b":   "Kredit boshqarilishi mumkin, ammo ayrim oylar qiyin bo'lishi mumkin.",
    "r_high_h":  "🔶 Yuqori yuklama — shartlarni qayta ko'rib chiqing",
    "r_high_b":  "Kredit byudjetingizning katta qismini egallaydi. Miqdorni kamaytiring yoki muddatni uzaytiring.",
    "r_crit_h":  "🔴 Kredit moliyaviy xavfli bo'lishi mumkin",
    "r_crit_b":  "Ma'lumotlaringizga ko'ra yuklama juda yuqori. Alternativalarni ko'rib chiqing: kam miqdor, uzoq muddat yoki kechiktirish. Bu moliyaviy maslahat emas.",
    "dist_h":"Kredit muddati bo'yicha yuklama","dist_low":"Oddiy oylar",
    "dist_med":"O'rtacha oylar","dist_high":"Qiyin oylar","dist_crit":"Kritik oylar",
    "rec_before_h":"Kreditni olishdan oldin","rec_reduce_h":"Yuklamani qanday kamaytirish",
    "rec_timing_h":"To'lov sanasi xavfi","rec_careful_h":"Ehtiyot bo'lish kerak bo'lgan hollar",
    "chk_date":  "To'lov kuni ish haqi kunidan oldin. Bankdan to'lov kunini ish haqidan keyinga ko'chirishni so'rang.",
    "chk_terms": "Shartlarni to'liq bilmaysiz. Bankdan so'rang: jami qiymat, komissiyalar, jarimalar.",
    "chk_total": "Kredit narxini to'liq bilib oling — nafaqat oylik to'lov, balki jami to'lanadigan summa.",
    "chk_buffer":"Imzolashdan oldin kamida 1–2 oylik to'lov miqdori jamg'armada bo'lsin.",
    "chk_delay": "Ish haqi kechiksa to'lovga qiynalasiz. Kamida bir oylik to'lov uchun zahira yarating.",
    "red_amount":"Kredit miqdorini kamaytiring — oylik to'lov kamayadi.",
    "red_term":  "Muddatni uzaytiring — oylik to'lov kamayadi (lekin jami ko'p bo'ladi).",
    "red_flex":  "Ixtiyoriy xarajatlarni vaqtincha kamaytiring.",
    "red_no_extra":"Bu kredit to'languncha yangi kredit olmang.",
    "tim_date":  "To'lov kuni ({p}) ish haqi kunidan ({s}) oldin. Bu har oy xavf yaratadi.",
    "tim_ok":    "To'lov kuni ({p}) ish haqidan ({s}) keyin — yaxshi. Sana xavfi minimal.",
    "car_salary":"Ish haqi kechikishi mumkin. Alohida hisobda kamida 2 oylik to'lov zahirasi saqlang.",
    "car_non_urg":"Kredit shoshilinch ehtiyoj uchun emas. Yuklama yuqori bo'lsa — kutish yoki kam olish yaxshiroq.",
    "car_refi":  "Yangi kredit eski kreditni to'lash uchun — qarz tuzoqchasi xavfi. Avval bank bilan qayta tuzishni muhokama qiling.",
    "car_low_inc":"Daromadingiz mamlakatdagi o'rtachadan past. To'lov byudjetning katta qismini egallaydi.",
    "car_macro": "Iqtisodiy sharoitlar qo'shimcha bosim yaratmoqda. Kamroq qarz olish yoki kutish tavsiya etiladi.",
    "car_flex":  "Ixtiyoriy xarajatlar ko'p. Ularni kamaytirish byudjetni bo'shatishi mumkin.",
    "exp_charts":"Oyma-oy batafsil prognoz","exp_table":"Simulyatsiya jadvali",
    "exp_macro": "Hisob-kitobda ishlatiladigan iqtisodiy parametrlar","exp_method":"Natija qanday hisoblanadi",
    "ch1_title": "Daromadning kredit to'loviga ketadigan ulushi — oylar bo'yicha",
    "ch2_title": "To'lovlardan keyingi qoldiq — oylar bo'yicha",
    "ml_proba_title": "Model ishonchi (Random Forest) — oylar bo'yicha bosim ehtimolligi",
    "macro_note":"Bu parametrlar faqat ichki hisob-kitob uchun ishlatiladi.",
    "download":  "Jadvalni yuklab olish (CSV)",
    "method_body":"""**Oylik to'lov** annuitet formulasi yordamida hisoblanadi.

**Makro bosim darajasi** o'qitilgan **Random Forest** modeli (200 daraxt, class_weight=balanced) tomonidan aniqlanadi. Model O'zbekistonning oylik makroiqtisodiy ma'lumotlari asosida o'qitilgan (2010–2025, 192 kuzatuv). Belgilar: inflyatsiya, qayta moliyalashtirish stavkasi, CPI MoM, nominal ish haqi, real ish haqi o'sishi, real stavka, qarz ko'rsatkichi. Maqsad: Low / Medium / High.

Model ko'rsatkichlari: Random Forest test aniqlik 92.3%, F1-macro 0.651. Decision Tree: aniqlik 79.5%, F1-macro 0.760.

**Oylik yuklama** quyidagilarga asoslanadi:
- To'lovning daromaddagi ulushi
- Barcha xarajatlar va to'lovlardan keyingi qoldiq
- Jamg'arma zahirasi
- To'lov va ish haqi sanasining nisbati

**Yuklama darajalari:**
- ✅ Oddiy — to'lov daromadning 25% gacha, zahira bor
- ⚠️ O'rtacha — 25–40%, ba'zi oylar qiyin bo'lishi mumkin
- 🔶 Yuqori — 40–50%, shartlarni qayta ko'rib chiqing
- 🔴 Kritik — 50% dan yuqori, yuqori moliyaviy xavf

Vosita bank qarorlarini qabul qilmaydi.""",
    "lbl_low":"Oddiy","lbl_med":"O'rtacha","lbl_high":"Yuqori","lbl_crit":"Kritik",
    # table columns
    "tbl_date":"Sana","tbl_month":"Oy","tbl_pmt":"To'lov","tbl_inc":"Daromad",
    "tbl_ess":"Majburiy","tbl_flex":"Ixtiyoriy","tbl_pti":"PTI %","tbl_tdb":"TDB %",
    "tbl_cash":"Qoldiq","tbl_bal":"Balans","tbl_lvl":"Daraja",
    # custom term
    "term_custom":"O'z muddati (oy)","term_mode_pre":"Ro'yxatdan tanlash","term_mode_own":"Qo'lda kiritish",
    # validation
    "v_rate_high":"Foiz stavkasi 100% dan yuqori — iltimos, tekshiring.",
    "v_exp_high":"Xarajatlar daromaddan oshib ketdi — ma'lumotlarni tekshiring.",
    "v_savings_low":"Jamg'arma bir oylik to'lovdan kam. Ariza berishdan oldin zahira yaratishni tavsiya etamiz.",
    # AI features
    # Monte Carlo ML model
    "mc_default_lbl":   "Default ehtimolligi (ML)",
    "mc_default_sub":   "Monte Carlo · Random Forest",
    "mc_no_model":      "loan_risk_model.pkl topilmadi. Avval model_training.py ni lokal ishga tushiring.",
    # Smart insights
    "si_title":   "🧠 Shaxsiy tahlil",
    "si_empty":   "Muammolar topilmadi — vaziyat normal ko'rinadi.",
    "si_critical":"Kritik","si_warning":"Diqqat","si_tip":"Maslahat","si_ok":"Yaxshi",
    # errors
    "err_dataset":"data_fw.xlsx topilmadi. Uni app.py bilan bir xil GitHub papkasiga joylashtiring.",
    "err_forecast":"Prognoz tuzilmadi. data_fw.xlsx ni tekshiring.",
    "disclaimer":"Tadqiqot vositasi. Moliyaviy maslahat, kredit skoringi yoki bank qarori emas.",
},
}  # end TR

# ══════════════════════════════════════════════════════════════
# CSS — blue primary, clean, minimal
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Layout ── */
.main>.block-container{max-width:760px;padding-top:.5rem;padding-bottom:3rem;}

/* ── Hero ── */
.hero{background:linear-gradient(135deg,#1d4ed8 0%,#2563eb 60%,#60a5fa 100%);
  border-radius:20px;padding:22px 28px 18px;color:#fff;margin-bottom:16px;}
.hero h1{font-size:1.55rem;font-weight:850;margin:0 0 4px;letter-spacing:-.03em;}
.hero p{margin:0;opacity:.9;font-size:.92rem;}

/* ── Welcome card ── */
.wcard{background:#fff;border:1px solid #dbeafe;border-radius:18px;
  padding:20px 24px;margin-bottom:14px;box-shadow:0 3px 14px rgba(15,23,42,.06);}

/* ── Wizard step bar ── */
.wbar{display:flex;margin-bottom:20px;border-radius:14px;overflow:hidden;border:1px solid #e2e8f0;}
.wi{flex:1;padding:9px 4px;text-align:center;font-size:.71rem;font-weight:750;
  background:#f8fafc;color:#94a3b8;border-right:1px solid #e2e8f0;}
.wi:last-child{border-right:none;}
.wi-done{background:#dbeafe;color:#1d4ed8;}
.wi-active{background:#2563eb;color:#fff;}

/* ── Input hint ── */
.hint{font-size:.73rem;color:#2563eb;font-weight:600;margin-top:-6px;
  margin-bottom:8px;font-family:'SF Mono',monospace;min-height:15px;letter-spacing:.01em;}

/* ── Budget preview ── */
.bcard{background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;
  padding:14px 18px;margin:10px 0;}
.brow{display:flex;justify-content:space-between;padding:5px 0;
  border-bottom:1px solid #f1f5f9;font-size:.85rem;}
.brow:last-child{border-bottom:none;padding-top:8px;margin-top:2px;font-weight:800;font-size:.9rem;}
.blbl{color:#64748b;}
.bval{font-weight:650;font-family:monospace;color:#0f172a;}
.bneg{color:#dc2626 !important;}
.bblue{color:#2563eb;}

/* ── Result boxes ── */
.rbox-low{background:#f0fdf4;border:1.5px solid #86efac;border-radius:16px;
  padding:16px 20px;color:#166534;margin-bottom:10px;}
.rbox-med{background:#fefce8;border:1.5px solid #fde047;border-radius:16px;
  padding:16px 20px;color:#854d0e;margin-bottom:10px;}
.rbox-high{background:#fff7ed;border:1.5px solid #fb923c;border-radius:16px;
  padding:16px 20px;color:#9a3412;margin-bottom:10px;}
.rbox-crit{background:#fef2f2;border:1.5px solid #f87171;border-radius:16px;
  padding:16px 20px;color:#991b1b;margin-bottom:10px;}

/* ── Metric cards ── */
.mcard{background:#fff;border:1px solid #e2e8f0;border-radius:14px;
  padding:12px 16px;text-align:center;box-shadow:0 2px 8px rgba(15,23,42,.04);}
.mlbl{color:#64748b;font-size:.76rem;font-weight:650;margin-bottom:3px;}
.mval{color:#0f172a;font-size:1.1rem;font-weight:850;letter-spacing:-.02em;}

/* ── Pressure pills ── */
.p-low{display:inline-block;padding:3px 11px;border-radius:999px;font-weight:750;
  font-size:.8rem;background:#f0fdf4;color:#166534;border:1px solid #86efac;}
.p-med{display:inline-block;padding:3px 11px;border-radius:999px;font-weight:750;
  font-size:.8rem;background:#fefce8;color:#854d0e;border:1px solid #fde047;}
.p-high{display:inline-block;padding:3px 11px;border-radius:999px;font-weight:750;
  font-size:.8rem;background:#fff7ed;color:#9a3412;border:1px solid #fb923c;}
.p-crit{display:inline-block;padding:3px 11px;border-radius:999px;font-weight:750;
  font-size:.8rem;background:#fef2f2;color:#991b1b;border:1px solid #f87171;}

/* ── Rec section label ── */
.rech{font-size:.76rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;
  color:#94a3b8;margin:16px 0 5px;}

/* ── Warning / info ── */
.wbox{background:#fefce8;border:1px solid #fde047;border-radius:12px;
  padding:10px 14px;color:#713f12;font-size:.87rem;margin-bottom:10px;}
.ibox{background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;
  padding:10px 14px;color:#1e3a8a;font-size:.87rem;margin-bottom:10px;}

/* ── Disclaimer ── */
.disc{background:#f8fafc;border:1px dashed #cbd5e1;border-radius:12px;
  padding:10px 14px;font-size:.76rem;color:#64748b;margin-top:1.2rem;}

/* ── Override Streamlit button colors: primary → blue ── */
.stButton>button[kind="primary"] {
    background-color: #2563eb !important;
    border-color: #2563eb !important;
    color: white !important;
}
.stButton>button[kind="primary"]:hover {
    background-color: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════
def ss(k, v):
    if k not in st.session_state: st.session_state[k] = v

ss("step",     0)
ss("sim_done", False)
ss("result",   None)
ss("lang",     "ru")

# ── Language selector ─────────────────────────────────────────
lc, _ = st.columns([2, 8])
with lc:
    sel_lang = st.selectbox(
        "🌐", list(LANG_OPTIONS.keys()),
        index=list(LANG_OPTIONS.values()).index(st.session_state["lang"]),
        label_visibility="collapsed", key="lang_sel",
    )
LANG = LANG_OPTIONS[sel_lang]
st.session_state["lang"] = LANG

def t(k: str) -> str:
    return TR.get(LANG, TR["ru"]).get(k, TR["ru"].get(k, k))

def pill(level: str) -> str:
    css = {"Low":"p-low","Medium":"p-med","High":"p-high","Critical":"p-crit"}.get(level,"p-med")
    lbl = {"Low":t("lbl_low"),"Medium":t("lbl_med"),"High":t("lbl_high"),"Critical":t("lbl_crit")}.get(level,level)
    return f'<span class="{css}">{lbl}</span>'

# ── Hero ──────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>{t('hero_title')}</h1>
  <p>{t('hero_sub')}</p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CORE HELPERS
# ══════════════════════════════════════════════════════════════
def fmt(v: float) -> str:
    """Format as spaced thousands + UZS."""
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"{abs(v):,.0f}".replace(",", " ") + " UZS"

def hint(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ") + " UZS" if v and v > 0 else ""

def pct(v: float, d: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"{v:.{d}f}%"

def ann(p: float, r_pct: float, m: int) -> float:
    if p <= 0 or m <= 0: return 0.0
    r = r_pct / 100 / 12
    if r == 0: return p / m
    return p * r * (1+r)**m / ((1+r)**m - 1)

def mcard(label: str, value: str, sub: str = "") -> None:
    sub_html = f"<div style='font-size:.72rem;color:#94a3b8;margin-top:2px'>{sub}</div>" if sub else ""
    st.markdown(f"""
<div class="mcard">
  <div class="mlbl">{label}</div>
  <div class="mval">{value}</div>
  {sub_html}
</div>""", unsafe_allow_html=True)

def wbar(active: int) -> str:
    # steps: 0=welcome (hidden), 1,2,3,4
    steps = [t("s1"), t("s2"), t("s3"), t("s4")]
    html  = '<div class="wbar">'
    for i, lbl in enumerate(steps, 1):
        if i < active:    c = "wi wi-done"; pfx = "✓ "
        elif i == active: c = "wi wi-active"; pfx = ""
        else:             c = "wi"; pfx = ""
        html += f'<div class="{c}">{pfx}{lbl}</div>'
    html += '</div>'
    return html

# ══════════════════════════════════════════════════════════════
# DATA LOADING — silent, cached
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError("data_fw.xlsx not found.")
    xls   = pd.ExcelFile(DATA_FILE)
    sheet = MAIN_SHEET if MAIN_SHEET in xls.sheet_names else xls.sheet_names[0]
    df    = pd.read_excel(DATA_FILE, sheet_name=sheet)
    df.columns = (df.columns.astype(str).str.strip()
                  .str.replace(" ","_",regex=False).str.replace("-","_",regex=False))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing: raise ValueError(f"Dataset missing columns: {missing}")
    df = df.dropna(how="all").copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["repayment_pressure_level"] = df["repayment_pressure_level"].astype(str).str.strip()
    df = df[df["repayment_pressure_level"].isin(["Low","Medium","High"])].copy()
    if "year"  not in df.columns: df["year"]  = df["date"].dt.year
    if "month" not in df.columns: df["month"] = df["date"].dt.month
    return df.reset_index(drop=True)

try:
    macro_df = load_data()
except Exception as e:
    st.error(t("err_dataset")); st.stop()

# ══════════════════════════════════════════════════════════════
# ML MODEL — train RandomForest on historical macro data
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def train_model(_df: pd.DataFrame):
    """Train RandomForestClassifier on the historical macro dataset.
    Uses the same 7 features as the dissertation analysis.
    Returns (clf, le) where le is the fitted LabelEncoder.
    """
    X = _df[FEATURE_COLUMNS].copy()
    for c in FEATURE_COLUMNS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median())
    y = _df["repayment_pressure_level"].astype(str).str.strip()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = RandomForestClassifier(
        n_estimators=200, random_state=42,
        class_weight="balanced", max_depth=6,
    )
    clf.fit(X.values, y_enc)
    return clf, le

def ml_pressure(row, clf, le) -> str:
    """Predict macro pressure level using the trained RF model."""
    feat = np.array([[float(row.get(c, 0) or 0) for c in FEATURE_COLUMNS]])
    pred = clf.predict(feat)[0]
    return le.inverse_transform([pred])[0]

def ml_pressure_proba(row, clf, le) -> dict[str, float]:
    """Return {class: probability} dict from the RF model."""
    feat = np.array([[float(row.get(c, 0) or 0) for c in FEATURE_COLUMNS]])
    proba = clf.predict_proba(feat)[0]
    return {le.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(proba)}

try:
    clf, le = train_model(macro_df)
except Exception as e:
    st.error(f"Model training failed: {e}"); st.stop()

# ══════════════════════════════════════════════════════════════
# MONTE CARLO DEFAULT RISK MODEL
# ══════════════════════════════════════════════════════════════
_MC_FEATURES = [
    "income", "loan_amount", "term_months", "interest_rate",
    "essential_exp", "flex_exp", "existing_loans", "savings",
    "unstable", "timing_risk",
]

@st.cache_resource(show_spinner=False)
def load_mc_model():
    """Load the Monte Carlo trained loan_risk_model.pkl if it exists."""
    if not _JOBLIB_AVAILABLE:
        return None
    try:
        payload = _joblib.load(MC_MODEL_FILE)
        return payload  # dict: {model, features, version, ...}
    except Exception:
        return None

def mc_default_probability(u, unstable: bool = False, timing_risk: bool = False) -> float | None:
    """Return P(default) from the Monte Carlo RF model, or None if unavailable."""
    mc = load_mc_model()
    if mc is None:
        return None
    try:
        X = pd.DataFrame([{
            "income":        u.income,
            "loan_amount":   u.amount,
            "term_months":   u.months,
            "interest_rate": u.rate,
            "essential_exp": u.ess,
            "flex_exp":      u.flex,
            "existing_loans":u.exist,
            "savings":       u.savings,
            "unstable":      float(not u.stable),
            "timing_risk":   float(u.pay_day < u.sal_day),
        }])
        cols = mc.get("features", _MC_FEATURES)
        proba = mc["model"].predict_proba(X[cols])[0]
        default_idx = list(mc["model"].classes_).index(1)
        return round(float(proba[default_idx]), 4)
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════
# MACRO FORECAST — cached by params
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def build_forecast(start_str: str, months: int, _clf, _le) -> pd.DataFrame:
    df = macro_df

    def damped(s: pd.Series, h: int, lo=None, hi=None) -> np.ndarray:
        c = pd.Series(s).astype(float).dropna().reset_index(drop=True)
        if c.empty: return np.zeros(h)
        w  = c.tail(min(36,len(c))).reset_index(drop=True)
        sl = 0.0
        if len(w)>=6:
            try: sl,_ = np.polyfit(np.arange(len(w)), w.values, 1)
            except Exception: pass
        sl *= 0.35
        fc = np.array([float(w.iloc[-1]) + sl*(i+1) for i in range(h)])
        if lo is not None: fc = np.maximum(fc, lo)
        if hi is not None: fc = np.minimum(fc, hi)
        return fc

    def sal_fc(s: pd.Series, h: int) -> np.ndarray:
        c = pd.Series(s).astype(float).dropna().reset_index(drop=True)
        if c.empty: return np.ones(h)*5_000_000
        g = c.pct_change().tail(24).replace([np.inf,-np.inf],np.nan).dropna()
        mg = float(np.clip(g.median() if not g.empty else 0.005, 0.001, 0.018))
        cur = float(c.iloc[-1])
        out: list[float] = []
        for _ in range(h): cur *= (1+mg); out.append(cur)
        return np.array(out)

    start = pd.to_datetime(start_str).to_period("M").to_timestamp()
    end   = start + pd.DateOffset(months=months-1)
    last  = df["date"].max().to_period("M").to_timestamp()

    fut_frames: list[pd.DataFrame] = []
    if end > last:
        fdt = pd.date_range(start=last+pd.DateOffset(months=1), end=end, freq="MS")
        fh  = len(fdt)
        fut = pd.DataFrame({"date": fdt})
        fut["policy_rate_pct"]             = damped(df["policy_rate_pct"],             fh, lo=0, hi=40)
        fut["inflation_yoy_pct"]           = damped(df["inflation_yoy_pct"],           fh, lo=0, hi=50)
        fut["cpi_mom_pct"]                 = damped(df["cpi_mom_pct"],                 fh, lo=-5, hi=10)
        fut["nominal_wage_monthly_approx"] = sal_fc(df["nominal_wage_monthly_approx"], fh)
        fut["debt_burden_indicator_pct"]   = damped(df["debt_burden_indicator_pct"],   fh, lo=0, hi=100)
        fut_frames.append(fut)

    hist = df[["date","policy_rate_pct","inflation_yoy_pct","cpi_mom_pct",
               "nominal_wage_monthly_approx","debt_burden_indicator_pct"]].copy()
    cb = pd.concat([hist]+fut_frames, ignore_index=True).sort_values("date")
    cb["yoy_nom"] = cb["nominal_wage_monthly_approx"].pct_change(12)*100
    cb["real_wage_growth_pct"]  = cb["yoy_nom"] - cb["inflation_yoy_pct"]
    cb["real_policy_rate_pct"]  = cb["policy_rate_pct"] - cb["inflation_yoy_pct"]
    last_rwg = float(df["real_wage_growth_pct"].dropna().iloc[-1])
    cb["real_wage_growth_pct"]  = cb["real_wage_growth_pct"].fillna(last_rwg)
    cb["real_policy_rate_pct"]  = cb["real_policy_rate_pct"].fillna(
        cb["policy_rate_pct"]-cb["inflation_yoy_pct"])
    cb["predicted_macro_pressure"] = cb.apply(
        lambda r: ml_pressure(r, _clf, _le), axis=1
    )
    # ML probability columns for each class
    proba_series = cb.apply(lambda r: ml_pressure_proba(r, _clf, _le), axis=1)
    cb["prob_Low"]    = proba_series.apply(lambda x: x.get("Low",    0.0))
    cb["prob_Medium"] = proba_series.apply(lambda x: x.get("Medium", 0.0))
    cb["prob_High"]   = proba_series.apply(lambda x: x.get("High",   0.0))

    fc = cb[(cb["date"]>=start)&(cb["date"]<=end)].copy()
    fc = fc.head(months).reset_index(drop=True)
    fc["loan_month"] = np.arange(1, len(fc)+1)
    return fc

# ══════════════════════════════════════════════════════════════
# USER INPUT DATACLASS
# ══════════════════════════════════════════════════════════════
@dataclass
class Inp:
    amount: float
    rate:   float
    months: int
    income: float
    ess:    float
    flex:   float
    exist:  float
    exist_mo: int
    savings:  float
    sal_day:  int
    pay_day:  int
    start_str:str
    proj:     str      # "avg"|"own"|"none"
    proj_pct: float
    infl_exp: bool
    stable:   bool
    purpose:  str
    knows:    bool
    can_delay:bool
    is_refi:  bool
    sav_mo:   int

# ══════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════
def simulate(u: Inp, fc: pd.DataFrame) -> dict:
    mp       = ann(u.amount, u.rate, u.months)
    inc      = float(u.income)
    ess      = float(u.ess)
    flex     = float(u.flex)
    bal      = float(u.savings)
    prev_sal = float(fc["nominal_wage_monthly_approx"].iloc[0])
    fg       = (1 + u.proj_pct/100)**(1/12) - 1
    rows: list[dict] = []

    for _, r in fc.iterrows():
        lm = int(r["loan_month"])
        if lm > 1:
            if u.proj == "avg":
                cs   = float(r["nominal_wage_monthly_approx"])
                g    = float(np.clip((cs/prev_sal-1) if prev_sal>0 else 0, -0.02, 0.03))
                inc *= (1+g); prev_sal = cs
            elif u.proj == "own":
                inc *= (1+fg)
            if u.infl_exp:
                cpi  = float(np.clip(float(r["cpi_mom_pct"])/100, -0.03, 0.05))
                ess  *= (1+cpi); flex *= (1+cpi)

        ex_pmt = u.exist if lm <= u.exist_mo else 0.0
        ttl    = ess + flex
        pti    = mp/inc if inc>0 else np.nan
        tdb    = (mp+ex_pmt)/inc if inc>0 else np.nan
        cash   = inc - ttl - ex_pmt - mp
        bal   += cash
        avgs   = float(r["nominal_wage_monthly_approx"])
        ivsa   = inc/avgs if avgs>0 else 1.0
        pb4    = u.pay_day < u.sal_day
        mp_lvl = r["predicted_macro_pressure"]
        p_low  = float(r.get("prob_Low",    0.0))
        p_med  = float(r.get("prob_Medium", 0.0))
        p_high = float(r.get("prob_High",   0.0))

        sc = 0
        if not np.isnan(pti):
            if pti>=0.40: sc+=3
            elif pti>=0.25: sc+=1
        if not np.isnan(tdb):
            if tdb>=0.50: sc+=3
            elif tdb>=0.35: sc+=1
        if cash<0: sc+=3
        elif inc>0 and cash<inc*0.10: sc+=1
        if bal<0: sc+=1
        if ivsa<0.80: sc+=1
        if pb4: sc+=1
        if mp_lvl=="High": sc+=2
        elif mp_lvl=="Medium": sc+=1
        if float(r["inflation_yoy_pct"])>=12: sc+=1
        if inc>0 and flex/inc>0.20: sc+=1

        if sc<=1: lv="Low"
        elif sc<=3: lv="Medium"
        elif sc<=6: lv="High"
        else: lv="Critical"

        rows.append({
            "date":r["date"],"month":lm,"payment":mp,
            "income":inc,"ess":ess,"flex":flex,"exist_pmt":ex_pmt,
            "pti_pct":pti*100 if not np.isnan(pti) else np.nan,
            "tdb_pct":tdb*100 if not np.isnan(tdb) else np.nan,
            "cash":cash,"balance":bal,
            "infl":r["inflation_yoy_pct"],"rate":r["policy_rate_pct"],
            "avg_sal":avgs,"macro":mp_lvl,
            "prob_low":p_low,"prob_med":p_med,"prob_high":p_high,
            "score":sc,"level":lv,
        })

    sim  = pd.DataFrame(rows)
    n    = len(sim)
    low  = int((sim["level"]=="Low").sum())
    med  = int((sim["level"]=="Medium").sum())
    high = int((sim["level"]=="High").sum())
    crit = int((sim["level"]=="Critical").sum())
    mc   = float(sim["cash"].min())
    mb   = float(sim["balance"].min())
    hs   = (high+crit)/n if n else 0
    cs   = crit/n if n else 0

    fh_s = sim[sim["level"].isin(["High","Critical"])]["month"]
    frm: int|None = int(fh_s.min()) if len(fh_s)>0 else None

    if cs>=0.15 or (crit>0 and mc<0) or mb<-u.income:  res="Critical"
    elif hs>=0.30 or mc<0 or mb<0:                       res="High"
    elif high>0 or (med/n>=0.50 if n else False):        res="Medium"
    else:                                                 res="Low"

    return {
        "sim":sim, "mp":round(mp,2),
        "total":round(mp*u.months,2),
        "low":low,"med":med,"high":high,"crit":crit,
        "avg_pti":round(float(sim["pti_pct"].mean()),2),
        "max_tdb":round(float(sim["tdb_pct"].max()),2),
        "min_cash":round(mc,2),
        "first_risky":frm, "result":res, "fc":fc,
        "default_prob": mc_default_probability(u),
    }



# ══════════════════════════════════════════════════════════════
# SMART INSIGHTS ENGINE — no API, purely from simulation data
# ══════════════════════════════════════════════════════════════
def _find_term_for_pti(inp, target_pti: float = 0.28):
    """Find shortest term (months) that brings monthly payment under target PTI."""
    if inp.income <= 0:
        return None, None
    for m in range(inp.months + 6, min(inp.months * 4, 360) + 1, 6):
        mp = ann(inp.amount, inp.rate, m)
        if mp / inp.income <= target_pti:
            return m, round(mp)
    return None, None

def _find_amount_for_pti(inp, target_pti: float = 0.28):
    """Find max loan amount keeping PTI under target at current term."""
    if inp.income <= 0 or inp.months <= 0:
        return None
    target_pmt = inp.income * target_pti
    r = inp.rate / 100 / 12
    if r == 0:
        amount = target_pmt * inp.months
    else:
        amount = target_pmt * ((1 + r)**inp.months - 1) / (r * (1 + r)**inp.months)
    rounded = round(amount / 500_000) * 500_000
    return rounded if rounded < inp.amount * 0.97 else None

def generate_smart_insights(R: dict, inp, lang: str) -> list[dict]:
    """
    Analyse simulation results and return ranked list of insights.
    Each dict: {level, icon, title, body}
    Levels: 'critical' | 'warning' | 'tip' | 'ok'
    Pure Python — no API calls.
    """
    sim     = R["sim"]
    mp      = R["mp"]
    avg_pti = R["avg_pti"]
    min_cash= R["min_cash"]
    n       = len(sim)
    crit_mo = R["crit"]
    high_mo = R["high"]
    dp      = R.get("default_prob")

    def fz(v):  return f"{abs(v):,.0f}".replace(",", " ") + " UZS"
    def fp(v):  return f"{v:.1f}%"
    def I(ru, en, uz): return {"ru": ru, "en": en, "uz": uz}[lang]

    insights: list[dict] = []

    # ── CRITICAL ───────────────────────────────────────────────
    if min_cash < 0:
        worst = int(sim["cash"].idxmin()) + 1
        insights.append({"level":"critical","icon":"🔴",
            "title": I(f"В месяц {worst} денег не хватит",
                       f"Cash runs out in month {worst}",
                       f"{worst}-oyda pul yetmaydi"),
            "body":  I(f"Минимальный остаток: −{fz(abs(min_cash))}. Платёж невозможно сделать без дополнительных источников.",
                       f"Minimum free cash: −{fz(abs(min_cash))}. Payment cannot be made without extra funds.",
                       f"Minimal qoldiq: −{fz(abs(min_cash))}. Qo'shimcha manbasisiz to'lovni amalga oshirib bo'lmaydi.")})

    if avg_pti > 50:
        insights.append({"level":"critical","icon":"🔴",
            "title": I(f"Больше половины дохода на кредит ({fp(avg_pti)})",
                       f"Over half of income goes to loan ({fp(avg_pti)})",
                       f"Daromadning yarmidan ko'pi kreditga ({fp(avg_pti)})"),
            "body":  I(f"Каждый месяц {fz(mp)} из {fz(inp.income)} дохода — только на кредит. Мировой стандарт: максимум 30–35%.",
                       f"Every month {fz(mp)} of {fz(inp.income)} income goes to the loan. Global standard: max 30–35%.",
                       f"Har oy {fz(inp.income)} daromaddan {fz(mp)} faqat kreditga. Global standart: maksimum 30–35%.")})

    if dp is not None and dp > 0.60:
        insights.append({"level":"critical","icon":"🔴",
            "title": I(f"Высокий риск дефолта по ML ({dp*100:.0f}%)",
                       f"High ML default risk ({dp*100:.0f}%)",
                       f"ML bo'yicha yuqori default xavfi ({dp*100:.0f}%)"),
            "body":  I(f"Из 100 000 смоделированных заёмщиков с похожим профилем {dp*100:.0f}% не смогли обслуживать такой кредит до конца срока.",
                       f"Of 100,000 simulated borrowers with a similar profile, {dp*100:.0f}% could not service this loan to the end.",
                       f"Shunga o'xshash profilga ega 100,000 ta simulyatsiya qilingan qarz oluvchidan {dp*100:.0f}% kreditni to'lay olmadi.")})

    # ── WARNINGS ───────────────────────────────────────────────
    if 30 < avg_pti <= 50:
        insights.append({"level":"warning","icon":"⚠️",
            "title": I(f"Нагрузка на доход {fp(avg_pti)} — выше нормы",
                       f"Income burden {fp(avg_pti)} — above comfort zone",
                       f"Daromad yuklamasi {fp(avg_pti)} — normadan yuqori"),
            "body":  I(f"Платёж {fz(mp)} составляет {fp(avg_pti)} дохода. Рекомендуемый максимум — 30%. При снижении дохода будет тяжело.",
                       f"Payment {fz(mp)} is {fp(avg_pti)} of income. Recommended max is 30%. Any income drop will be difficult.",
                       f"To'lov {fz(mp)} daromadning {fp(avg_pti)}ini tashkil etadi. Tavsiya etilgan maksimum — 30%.")})

    if inp.savings < mp * 2:
        sav_mo = inp.savings / mp if mp > 0 else 0
        needed = mp * 3
        insights.append({"level":"warning","icon":"⚠️",
            "title": I(f"Подушка только на {sav_mo:.1f} месяца",
                       f"Savings cover only {sav_mo:.1f} months",
                       f"Jamg'arma faqat {sav_mo:.1f} oyni qoplaydi"),
            "body":  I(f"Сбережения {fz(inp.savings)} — это {sav_mo:.1f} платежа. Рекомендуется минимум 3 платежа ({fz(needed)}) до оформления.",
                       f"Savings {fz(inp.savings)} = {sav_mo:.1f} payments. Recommended minimum before signing: 3 payments ({fz(needed)}).",
                       f"Jamg'arma {fz(inp.savings)} = {sav_mo:.1f} to'lov. Imzolashdan oldin tavsiya etilgan minimum: 3 to'lov ({fz(needed)}).")})

    if inp.pay_day < inp.sal_day:
        gap = inp.sal_day - inp.pay_day
        insights.append({"level":"warning","icon":"⚠️",
            "title": I(f"Платёж {inp.pay_day}-го раньше зарплаты {inp.sal_day}-го",
                       f"Payment day {inp.pay_day} is before salary day {inp.sal_day}",
                       f"To'lov kuni {inp.pay_day} ish haqi kunidan {inp.sal_day} oldin"),
            "body":  I(f"Каждый месяц разрыв {gap} дней — деньги ещё не поступили. Попросите банк перенести дату платежа.",
                       f"Every month a {gap}-day gap — money has not arrived yet. Ask the bank to move the payment date.",
                       f"Har oy {gap} kunlik bo'shliq — pul hali kelmagan. Bankdan to'lov kunini o'tkazishni so'rang.")})

    if inp.is_refi:
        insights.append({"level":"warning","icon":"⚠️",
            "title": I("Кредит для погашения другого — риск долговой ловушки",
                       "Loan to pay off another loan — debt trap risk",
                       "Boshqa kreditni to'lash uchun kredit — qarz tuzoqchasi xavfi"),
            "body":  I("Рефинансирование увеличивает общую переплату. Сначала обсудите с банком реструктуризацию — часто это выгоднее.",
                       "Refinancing increases total repayment. First discuss restructuring with your bank — often a better option.",
                       "Qayta moliyalashtirish jami to'lovni oshiradi. Avval bank bilan qayta tuzishni muhokama qiling.")})

    heavy_pct = (crit_mo + high_mo) / n * 100 if n > 0 else 0
    if heavy_pct > 30:
        insights.append({"level":"warning","icon":"⚠️",
            "title": I(f"{heavy_pct:.0f}% месяцев кредита будут тяжёлыми",
                       f"{heavy_pct:.0f}% of loan months are high-pressure",
                       f"Kredit oylarining {heavy_pct:.0f}% og'ir bo'ladi"),
            "body":  I(f"Начиная с месяца {R['first_risky']}: {crit_mo+high_mo} из {n} месяцев прогнозируются с высокой нагрузкой.",
                       f"From month {R['first_risky']}: {crit_mo+high_mo} of {n} months forecast as high-burden.",
                       f"{R['first_risky']}-oydan boshlab: {n} oydan {crit_mo+high_mo} tasi yuqori yuklamali.")})

    # ── TIPS / ALTERNATIVES ────────────────────────────────────
    safe_term, safe_mp = _find_term_for_pti(inp)
    if safe_term and safe_term > inp.months:
        extra = safe_term - inp.months
        save_mo = round(mp - safe_mp)
        overpay = round((safe_mp * safe_term) - (mp * inp.months))
        insights.append({"level":"tip","icon":"💡",
            "title": I(f"Срок +{extra} мес. → платёж {fz(safe_mp)}/мес",
                       f"Term +{extra} months → payment {fz(safe_mp)}/month",
                       f"Muddat +{extra} oy → to'lov {fz(safe_mp)}/oy"),
            "body":  I(f"Увеличение срока до {safe_term} мес. снизит платёж с {fz(mp)} до {fz(safe_mp)}: экономия {fz(save_mo)}/мес. Общая переплата вырастет на {fz(overpay)}.",
                       f"Extending to {safe_term} months cuts payment from {fz(mp)} to {fz(safe_mp)}: saving {fz(save_mo)}/month. Total repayment increases by {fz(overpay)}.",
                       f"Muddatni {safe_term} oyga uzaytirish to'lovni {fz(mp)} dan {fz(safe_mp)} ga tushiradi: oyiga {fz(save_mo)} tejash. Jami to'lov {fz(overpay)} ga oshadi.")})

    safe_amt = _find_amount_for_pti(inp)
    if safe_amt:
        reduction = round(inp.amount - safe_amt)
        safe_mp_a = round(ann(safe_amt, inp.rate, inp.months))
        insights.append({"level":"tip","icon":"💡",
            "title": I(f"Сумма −{fz(reduction)} → PTI под 28%",
                       f"Amount −{fz(reduction)} → PTI under 28%",
                       f"Miqdor −{fz(reduction)} → PTI 28% dan past"),
            "body":  I(f"Уменьшение суммы до {fz(safe_amt)} снизит платёж с {fz(mp)} до {fz(safe_mp_a)}/мес — без увеличения срока.",
                       f"Reducing to {fz(safe_amt)} cuts payment from {fz(mp)} to {fz(safe_mp_a)}/month — without extending the term.",
                       f"Miqdorni {fz(safe_amt)} ga kamaytirish to'lovni {fz(mp)} dan {fz(safe_mp_a)}/oyga tushiradi — muddatni uzaytirmasdan.")})

    if inp.flex > mp * 0.3:
        freed = round(inp.flex * 0.5)
        insights.append({"level":"tip","icon":"💡",
            "title": I(f"Сократить необязательные расходы → +{fz(freed)}/мес",
                       f"Cut optional spending → +{fz(freed)}/month",
                       f"Ixtiyoriy xarajatlarni kamaytirish → +{fz(freed)}/oy"),
            "body":  I(f"У вас {fz(inp.flex)}/мес необязательных расходов. Временное сокращение вдвое освободит {fz(freed)}/мес — запас для тяжёлых месяцев.",
                       f"You have {fz(inp.flex)}/month in optional expenses. Temporarily halving them frees {fz(freed)}/month — a buffer for tough months.",
                       f"Ixtiyoriy xarajatlaringiz {fz(inp.flex)}/oy. Vaqtincha ikki baravar kamaytirish {fz(freed)}/oy bo'shatadi.")})

    # ── POSITIVE SIGNALS ───────────────────────────────────────
    if inp.pay_day >= inp.sal_day:
        insights.append({"level":"ok","icon":"✅",
            "title": I("Дата платежа после зарплаты — хорошо",
                       "Payment day is after salary — good",
                       "To'lov kuni ish haqidan keyin — yaxshi"),
            "body":  I(f"Платёж {inp.pay_day}-го после зарплаты {inp.sal_day}-го. Риск технической просрочки минимальный.",
                       f"Payment on day {inp.pay_day} after salary on day {inp.sal_day}. Late payment risk is minimal.",
                       f"{inp.pay_day}-kunda to'lov {inp.sal_day}-kundagi ish haqidan keyin. Kechikish xavfi minimal.")})

    if avg_pti <= 25 and min_cash >= 0:
        insights.append({"level":"ok","icon":"✅",
            "title": I("Нагрузка умеренная — кредит управляемый",
                       "Moderate load — loan is manageable",
                       "Yuklama o'rtacha — kredit boshqarilishi mumkin"),
            "body":  I(f"PTI {fp(avg_pti)} ниже 25%. Даже в худший месяц остаётся {fz(min_cash)}.",
                       f"PTI {fp(avg_pti)} is under 25%. Even in the worst month {fz(min_cash)} remains.",
                       f"PTI {fp(avg_pti)} — 25% dan past. Eng og'ir oyda ham {fz(min_cash)} qoladi.")})

    if inp.exist == 0:
        insights.append({"level":"ok","icon":"✅",
            "title": I("Нет других кредитов — хороший старт",
                       "No existing loans — good starting point",
                       "Boshqa kreditlar yo'q — yaxshi boshlang'ich"),
            "body":  I("Отсутствие текущих кредитных обязательств снижает общую нагрузку и даёт финансовую гибкость.",
                       "No current loan obligations reduces total burden and gives financial flexibility.",
                       "Joriy kredit majburiyatlarining yo'qligi umumiy yuklamani kamaytiradi.")})

    # Sort: critical first, then warnings, tips, ok
    order = {"critical":0,"warning":1,"tip":2,"ok":3}
    insights.sort(key=lambda x: order.get(x["level"], 9))
    return insights

# ══════════════════════════════════════════════════════════════
