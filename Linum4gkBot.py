from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes,
    ConversationHandler, MessageHandler, filters
)
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import CallbackQueryHandler
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from io import BytesIO
from joblib import load
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

PERFORMANCE = range(1)

# === Shared Selenium driver and buffer ===
driver = None
buffer = None
histogram_cache = {}

# === Load buffer.csv ===
if os.path.exists("buffer.csv"):
    buffer = pd.read_csv("buffer.csv")
else:
    buffer = pd.DataFrame(columns=[
        'Percent', 'Count', 'TrueDL', 'Weekend', 'CHOlg', 'LOlg', 'PDm', 'OtherPlrs', 'Total_Questions',
        'овсч', 'кубок эквестрии', 'балтийский бриз', 'островок бесконечности',
        'бесконечные земли', 'лёгкий смоленск', 'простой смоленск', 'куб',
        'editors', 'Difficulty_est','ID'
    ])

# === Authorization flag ===
WHO_PLAYS = range(1)

# Список доступных игроков
PLAYERS_LIST = [
    "Кирилл Юдин", "Ольга Лиманская", "Ольга Чуваткина", "Андрей Коссенков", "Василий Ковалёв", "Ксения Ковалёва",
    "Дмитрий Переладов", "Ксения Чурюмова", "Владислав Мельников", "Иной игрок"]

# Временное хранилище выбранных игроков
selected_players = {}

pending_ids = {}  # user_id -> tournament_id

def is_allowed(user_id):
    return user_id in ALLOWED_USERS

def parse_tournament_data(tournament_id):
    global driver
    url = f"https://rating.chgk.info/tournament/{tournament_id}"
    driver.get(url)
    time.sleep(3)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "teams_table_new"))
    )

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Название турнира
    title_elem = soup.find("h1")
    tournament_name = title_elem.get_text(strip=True) if title_elem else f"Tournament {tournament_id}"

    # Сложность и TrueDL

    difficulty_block = soup.find(string=lambda s: s and "Сложность заявленная" in s)
    declared_difficulty = "N/A"
    true_dl = "N/A"

    if difficulty_block:
        strong = difficulty_block.find_next("strong")
        if strong:
            parts = strong.text.strip().split("/")
            if len(parts) == 2:
                declared_difficulty = parts[0].strip()
                true_dl_raw = parts[1].strip()
                true_dl = true_dl_raw if true_dl_raw != "-" else "N/A"
            elif len(parts) == 1:
                declared_difficulty = parts[0].strip()

    # Редакторы
    editors = []
    editors_section = soup.find("div", class_="card", string=lambda t: t and "Редакторы" in t)
    if not editors_section:
        all_cards = soup.find_all("div", class_="card")
        if len(all_cards) > 1:
            editors_section = all_cards[1]
    if editors_section:
        editors = [
            s.get_text(strip=True).split("–")[0].strip()
            for s in editors_section.find_all("span", class_="card_row")
        ]

    # Таблица
    table = soup.find("table", {"id": "teams_table_new"})
    headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
    rows = [
        [td.get_text(strip=True) for td in tr.find_all("td")]
        for tr in table.find("tbody").find_all("tr")
    ]
    df = pd.DataFrame(rows, columns=headers[:len(rows[0])])
    values = pd.to_numeric(df["Взято"].str.replace(',', '.'), errors='coerce').dropna()

    histogram_cache[tournament_id] = values
    
    return values, tournament_name, declared_difficulty, true_dl, editors

def create_histogram(values, bins, bin_labels, title, highlight=None):
    plt.figure(figsize=(8, 4))
    counts, bins_edges, patches = plt.hist(values, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel("Взято")
    plt.ylabel("Количество команд")

    mid = 0.5 * (np.array(bins[:-1]) + np.array(bins[1:]))
    plt.xticks(mid, bin_labels, rotation=45)

    for count, x in zip(counts, mid):
        if count > 0:
            plt.text(x, count + 0.5, str(int(count)), ha='center', va='bottom', fontsize=8)

    # Штрихуем нужную треть бина
    if highlight is not None:
        for i in range(len(bins) - 1):
            left, right = bins[i], bins[i + 1]
            if left <= highlight < right:
                bin_width = right - left
                third = bin_width / 3
                height = counts[i] if i < len(counts) else 0

                # Вычисляем треть, куда попадает highlight
                if highlight < left + third:
                    shade_left = left
                elif highlight < left + 2 * third:
                    shade_left = left + third
                else:
                    shade_left = left + 2 * third

                # рисуем штриховку для 1/3 бина
                plt.bar(shade_left, height, width=third, align='edge',
                        color='red', alpha=0.5, hatch='//', edgecolor='red', linewidth=0)
                break  # нашли нужный бин, дальше не ищем

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

async def send_histograms_and_stats(update: Update, tournament_id: str):
    try:
        if tournament_id in histogram_cache:
            values = histogram_cache[tournament_id]
            # Получим остальную информацию всё равно — она не кэшируется пока
            _, tournament_name, difficulty, true_dl, editors = parse_tournament_data(tournament_id)
        else:
            values, tournament_name, difficulty, true_dl, editors = parse_tournament_data(tournament_id)

        target = update.message or update.callback_query.message

        bins_6 = [1, 7, 13, 19, 25, 31, 37]
        labels_6 = ["1–6", "7–12", "13–18", "19–24", "25–30", "31–36"]
        hist_6 = create_histogram(values, bins=bins_6, bin_labels=labels_6, title=f"{tournament_name} (6 кластеров)")
        await target.reply_photo(photo=InputFile(hist_6, filename="hist6.png"))

        bins_12 = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37]
        labels_12 = ["1–3", "4–6", "7–9", "10–12", "13–15", "16–18", "19–21", "22–24", "25–27", "28–30", "31–33", "34–36"]
        hist_12 = create_histogram(values, bins=bins_12, bin_labels=labels_12, title=f"{tournament_name} (12 кластеров)")
        await target.reply_photo(photo=InputFile(hist_12, filename="hist12.png"))

        median = np.median(values)
        await target.reply_text(f"Медиана: {median:.2f}")

        if editors:
            editor_list = "\n".join(f"— {e}" for e in editors)
        else:
            editor_list = "Unknown"

        await target.reply_text(f"Редакторы:\n{editor_list}")

    except Exception as e:
        target = update.message or update.callback_query.message
        await target.reply_text(f"❌ Ошибка при построении гистограмм: {e}")
    
async def select_players_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    #if not is_allowed(user_id):
    #    await update.message.reply_text("Я вас не знаю")
    #    return ConversationHandler.END

    if not context.args:
        await update.message.reply_text("Мне нужно ID турнира, желательно синхронная версия, e.g. /selectplayers 10183")
        return ConversationHandler.END

    pending_ids[user_id] = context.args[0]
    selected_players[user_id] = set()
    await send_player_keyboard(update, context, user_id)
    return WHO_PLAYS

async def send_player_keyboard(update_or_callback, context, user_id):
    selected = selected_players.get(user_id, set())

    keyboard = []
    row = []
    for i, player in enumerate(PLAYERS_LIST):
        prefix = "✅ " if player in selected else ""
        row.append(InlineKeyboardButton(prefix + player, callback_data=f"toggle:{player}"))

        if len(row) == 2:
            keyboard.append(row)
            row = []

    if row:
        keyboard.append(row)

    if len(selected) > 0:
        keyboard.append([InlineKeyboardButton("▶️ Завершить", callback_data="done")])

    text = "Кто играет?\n\n" + "\n".join("✅ " + p for p in selected)
    
    if hasattr(update_or_callback, "message") and update_or_callback.message:
        await update_or_callback.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    elif hasattr(update_or_callback, "callback_query"):
        await update_or_callback.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_player_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    data = query.data
    if data.startswith("toggle:"):
        player = data.split("toggle:")[1]
        selected = selected_players.get(user_id, set())
        if player in selected:
            selected.remove(player)
        else:
            if len(selected) >= 6:
                await query.edit_message_text("❌ Больше 6 игроков нельзя. Начни заново с /selectplayers <id>")
                return ConversationHandler.END
            selected.add(player)
        selected_players[user_id] = selected
        await send_player_keyboard(update, context, user_id)
        return WHO_PLAYS

    elif data == "done":
        if user_id not in pending_ids or user_id not in selected_players:
            await query.edit_message_text("Internal error.")
            return ConversationHandler.END

        player_list = selected_players.pop(user_id)
        tournament_id = pending_ids.pop(user_id)

        # Здесь вызови существующий parse_tournament_data и добавь строку в buffer
        await finalize_buffer_entry(update, context, user_id, tournament_id, player_list)

        return ConversationHandler.END

def detect_thematic_columns(tournament_name):
    thematic_flags = {
        "овсч": 0,
        "кубок эквестрии": 0,
        "балтийский бриз": 0,
        "островок бесконечности": 0,
        "бесконечные земли": 0,
        "лёгкий смоленск": 0,
        "простой смоленск": 0,
        "куб": 0
    }
    normalized_name = tournament_name.lower().replace("«", "").replace("»", "")
    keywords = {
        "овсч": "овсч",
        "кубок эквестрии": "эквестрии",
        "балтийский бриз": "бриз",
        "островок бесконечности": "островок",
        "бесконечные земли": "бесконечные земли",
        "лёгкий смоленск": "лёгкий смоленск",
        "простой смоленск": "простой смоленск",
        "куб": "куб"
    }
    for col, keyword in keywords.items():
        if keyword in normalized_name:
            thematic_flags[col] = 1
    return thematic_flags

def detect_players(player_list):
    players_flags = {
        "CHOlg": 0,
        "LOlg": 0,
        "PDm": 0,
        "OtherPlrs": 0
    }
    for player in player_list:
        if player == "Ольга Чуваткина":
            players_flags["CHOlg"] = 1
        elif player == "Ольга Лиманская":
            players_flags["LOlg"] = 1
        elif player == "Дмитрий Переладов":
            players_flags["PDm"] = 1
        else:
            players_flags["OtherPlrs"] += 1
    return players_flags

async def finalize_buffer_entry(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int, tournament_id: str, player_list: set):
    global buffer
    try:
        values, tournament_name, declared_difficulty, true_dl, editors = parse_tournament_data(tournament_id)

        new_row = {col: 0 for col in buffer.columns}
        new_row["ID"] = int(tournament_id)
        new_row["Difficulty_est"] = declared_difficulty
        new_row["TrueDL"] = true_dl
        new_row["Editors"] = "; ".join(editors)
        new_row["Total_Questions"] = 36
        new_row["Weekend"] = datetime.today().weekday() >= 5

        if int(tournament_id) in list(buffer["ID"]):
            count_value = buffer.loc[buffer["ID"] == int(tournament_id), "Count"].iloc[0]
        else:
            count_value = buffer["Count"].max() + 1 if not buffer.empty else 408
        new_row["Count"] = count_value

        #new_row["Count"] = buffer["Count"].max() if int(tournament_id) in list(buffer["ID"]) else buffer["Count"].max() + 1
        
        new_row["tournament_name"] = tournament_name

        # Заполняем тематические колонки
        thematic = detect_thematic_columns(tournament_name)
        for col, val in thematic.items():
            new_row[col] = val

        # Заполняем игроков
        players = detect_players(player_list)
        for col, val in players.items():
            new_row[col] = val

        buffer = pd.concat([pd.DataFrame([new_row]), buffer], ignore_index=True)
        buffer.to_csv("buffer.csv", index=False)

        await update.effective_message.reply_text("✅ Записал")
    except Exception as e:
        await update.effective_message.reply_text(f"❌ Error while saving: {e}")

    # === Предсказание Percent по новой строке ===
    try:
        pred_df = buffer.head(1).copy()

        # Заменим TrueDL на Difficulty_est, если он пустой или 'N/A'
        if pd.isna(pred_df["TrueDL"].iloc[0]) or pred_df["TrueDL"].iloc[0] == "N/A":
            pred_df["TrueDL"] = pred_df["Difficulty_est"]

        # Выбираем первые 15 признаков
        X_pred = pred_df.iloc[:, 0:16].copy()
        X_pred = X_pred.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Масштабируем и предсказываем
        X_scaled = scaler.transform(X_pred)
        percent_pred = model.predict(X_scaled)[0]

        # Записываем в колонку
        buffer.at[0, "Percent_predicted"] = percent_pred
        buffer.to_csv("buffer.csv", index=False)

        # Считаем предполагаемое количество взятых вопросов
        predicted_questions = round(percent_pred * pred_df["Total_Questions"].iloc[0], 1)

        await update.effective_message.reply_text(
            f"Предсказано: {predicted_questions} из 36 (на основе {percent_pred:.1%})"
        )
    except Exception as e:
        
        await update.effective_message.reply_text(f"⚠️ Ошибка при предсказании: {e}")

    await send_histograms_and_stats(update, tournament_id)        

async def performance_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global buffer
    
    try:
        if len(context.args) != 2:
            await update.message.reply_text("Введите ID турнира и количество взятых вопросов.\nПример: /performance 10183 27")
            return

        tid, correct = int(context.args[0]), float(context.args[1])
        row_index = buffer.index[buffer["ID"] == tid].tolist()

        if not row_index:
            await update.message.reply_text("❌ Турнир не был сыгран.")
            return

        idx = row_index[0]
        total = buffer.at[idx, "Total_Questions"]
        buffer.at[idx, "Percent"] = round(correct / total, 4)
        buffer.to_csv("buffer.csv", index=False)

        # Найдём значения "Взято" по текущему турниру
        from_tournament = buffer[buffer["ID"] == tid].copy()
        #values, tournament_name, *_ = parse_tournament_data(tid)

        if tid in histogram_cache:
            values = histogram_cache[tid]
            _, tournament_name, *_ = parse_tournament_data(tid)
        else:
            values, tournament_name, *_ = parse_tournament_data(tid)

        # Рисуем гистограмму с линией
        bins_12 = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37]
        labels_12 = ["1–3", "4–6", "7–9", "10–12", "13–15", "16–18", "19–21", "22–24",
             "25–27", "28–30", "31–33", "34–36"]

        histogram = create_histogram(values, bins=bins_12, bin_labels=labels_12,
                             title=f"Распределение результатов (ID {tid})", highlight=correct)

        await update.message.reply_photo(photo=InputFile(histogram, filename="perf_hist.png"))

        await update.message.reply_text(f"✅ Сохранено: {correct} из {total} → {correct / total:.1%}")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")

async def about_prediction_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global buffer
    try:
        if len(context.args) != 1:
            await update.message.reply_text("Введите ID турнира. Пример: /aboutprediction 10183")
            return

        tournament_id = int(context.args[0])
        matching_rows = buffer[buffer["ID"] == tournament_id]

        if matching_rows.empty:
            await update.message.reply_text("❌ Турнир с таким ID не найден")
            return

        row = matching_rows.head(1).copy()

        if pd.isna(row["TrueDL"].iloc[0]) or row["TrueDL"].iloc[0] == "N/A":
            row["TrueDL"] = row["Difficulty_est"]

        X = row.iloc[:, 0:16].apply(pd.to_numeric, errors='coerce').fillna(0)
        X_scaled = scaler.transform(X)

        feature_names = list(X.columns)
        feature_names = ["Сыграно игр" if name == "Count" else name for name in feature_names]
        contributions = model.coef_[:len(feature_names)] * X_scaled[0]

        #tournament_name = row.get("tournament_name", f"Турнир {tournament_id}")
        #if isinstance(tournament_name, pd.Series):
        #    tournament_name = tournament_name.iloc[0]

        tournament_name = row["tournament_name"].iloc[0] if "tournament_name" in row.columns else f"Турнир {tournament_id}"

        text = f"Вклад признаков в предсказание лассо регрессии для турнира '{tournament_name}':\n\n"

        players_sum = 0.0
        insignificant_total = 0.0
        for name, val in zip(feature_names, contributions):
            if name in ["CHOlg", "LOlg", "PDm", "OtherPlrs"]:
                players_sum += val
            elif abs(val) < 0.006:
                insignificant_total += val
            else:
                sign = "➕" if val > 0 else "➖"
                text += f"{sign} {name}: {val:.3f}\n"

        if players_sum != 0:
            sign = "➕" if players_sum > 0 else "➖"
            text += f"{sign} Состав: {players_sum:.3f}\n"

        if insignificant_total != 0:
            sign = "➕" if insignificant_total > 0 else "➖"
            text += f"{sign} Другие малозначимые: {insignificant_total:.3f}\n"

        total = contributions.sum() + model.intercept_
        text += f"\nСумма вклада: {total-model.intercept_:.3f}\nИнтерсепт модели: {model.intercept_:.3f}\nПроцент взятия: {total:.3f}"

        await update.message.reply_text(text)

    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка: {e}")

def setup_driver():
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(
    service=Service(ChromeDriverManager(version="114.0.5735.90").install()),
    options=options
)

    driver.get("https://rating.chgk.info/login")
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "_username")))
    driver.find_element(By.NAME, "_username").send_keys(EMAIL)
    driver.find_element(By.NAME, "_password").send_keys(PASSWORD)
    driver.find_element(By.ID, "login_go").click()
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//a[@href='/logout']")))

    print("Authentication successful")
    return driver

# === Telegram Bot Start ===
if __name__ == "__main__":
    driver = setup_driver()

    model = load("lasso_loocv_model.joblib")  # путь к модели
    scaler = load("scaler.joblib")            # путь к scaler
    
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    player_handler = ConversationHandler(
        entry_points=[CommandHandler("selectplayers", select_players_start)],
        states={WHO_PLAYS: [CallbackQueryHandler(handle_player_selection)]},
        fallbacks=[]
    )

    app.add_handler(player_handler)
    
    app.add_handler(CommandHandler("performance", performance_handler))

    app.add_handler(CommandHandler("aboutprediction", about_prediction_handler))
    
    print("Bot is running")
    app.run_polling()
