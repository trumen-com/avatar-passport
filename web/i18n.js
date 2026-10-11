// English / Russian / Uzbek (Latin) for the web UI. English is the source text: every text node, placeholder,
// aria-label, title and data-label that matches the dictionary (exactly, or by a pattern) is replaced, including text
// added later by app.js (a MutationObserver). Text without an entry stays in English.

export const LANGS = ["en", "ru", "uz"];

export function getLang() {
  try {
    const v = localStorage.getItem("avto-lang");
    if (LANGS.includes(v)) return v;
  } catch {
    /* storage blocked */
  }
  return "en";
}

export function setLang(lang) {
  try {
    localStorage.setItem("avto-lang", lang);
  } catch {
    /* storage blocked: the page still reloads in English */
  }
}

// English -> [Russian, Uzbek Latin]
const E = {
  // shell
  "Vehicle passport": ["Паспорт автомобиля", "Avtomobil pasporti"],
  "Car": ["Автомобиль", "Avtomobil"],
  "3D model": ["3D-модель", "3D model"],
  "Customize": ["Покраска", "Rang tanlash"],
  "Damage": ["Повреждения", "Shikastlar"],
  "Body shops": ["Автосервисы", "Ustaxonalar"],
  "VIN / History": ["VIN / История", "VIN / Tarix"],
  "Report": ["Отчёт", "Hisobot"],
  "New inspection": ["Новый осмотр", "Yangi ko‘rik"],
  "Share": ["Поделиться", "Ulashish"],
  "Share link": ["Поделиться ссылкой", "Havolani ulashish"],
  "Export passport": ["Скачать паспорт", "Pasportni yuklab olish"],
  "Condition passport": ["Паспорт состояния", "Holat pasporti"],
  "Turn a phone walkaround video into a verified condition passport.": ["Видео осмотра с телефона превращается в проверенный паспорт состояния.", "Telefondagi aylanma video tasdiqlangan holat pasportiga aylanadi."],
  "Checking server…": ["Проверка сервера…", "Server tekshirilmoqda…"],
  "Server unreachable": ["Сервер недоступен", "Server bilan aloqa yo‘q"],
  "Dark mode": ["Тёмная тема", "Qorong‘i rejim"],
  "Light mode": ["Светлая тема", "Yorug‘ rejim"],
  "Switch to dark mode": ["Включить тёмную тему", "Qorong‘i rejimga o‘tish"],
  "Switch to light mode": ["Включить светлую тему", "Yorug‘ rejimga o‘tish"],
  "Registry connector — sandbox (synthetic data)": ["Подключение к реестрам — песочница (синтетические данные)", "Reyestrlarga ulanish — sinov muhiti (sintetik ma’lumotlar)"],
  "Source code (AGPL-3.0)": ["Исходный код (AGPL-3.0)", "Manba kodi (AGPL-3.0)"],
  // capture
  "Record or upload": ["Снимите или загрузите", "Yozib oling yoki yuklang"],
  "Film the car all the way round (30–60 s), then photograph the VIN plate.": ["Снимите машину по кругу (30–60 с), затем сфотографируйте табличку VIN.", "Mashinani to‘liq aylanib suratga oling (30–60 s), so‘ng VIN raqamini suratga oling."],
  "Demo car": ["Демо-автомобиль", "Demo avtomobil"],
  "Use demo car": ["Использовать демо", "Demo mashinani ishlatish"],
  "1. Walkaround video": ["1. Видео по кругу", "1. Aylanma video"],
  "(required)": ["(обязательно)", "(majburiy)"],
  "Record walkaround": ["Снять видео", "Video yozish"],
  "Choose a video file": ["Выбрать видео", "Videoni tanlash"],
  "No video selected.": ["Видео не выбрано.", "Video tanlanmagan."],
  "2. VIN plate photo": ["2. Фото таблички VIN", "2. VIN raqami surati"],
  "(optional)": ["(необязательно)", "(ixtiyoriy)"],
  "Photograph VIN plate": ["Сфотографировать VIN", "VIN raqamini suratga olish"],
  "Choose a photo": ["Выбрать фото", "Suratni tanlash"],
  "No VIN photo selected.": ["Фото VIN не выбрано.", "VIN surati tanlanmagan."],
  "…or type the VIN instead": ["…или введите VIN вручную", "…yoki VIN raqamini yozing"],
  "17 characters (no I, O or Q)": ["17 символов (без I, O и Q)", "17 ta belgi (I, O, Q siz)"],
  "Leave empty to read the VIN from the photo.": ["Оставьте пустым, чтобы VIN прочитался с фото.", "VIN suratdan o‘qilishi uchun bo‘sh qoldiring."],
  "Tips for a good video": ["Советы для хорошего видео", "Yaxshi video uchun maslahatlar"],
  "Walk slowly round the car, 2–3 m away.": ["Медленно обойдите машину на расстоянии 2–3 м.", "Mashina atrofida 2–3 m masofada sekin yuring."],
  "Keep the whole car in frame.": ["Держите всю машину в кадре.", "Butun mashina kadrda bo‘lsin."],
  "Daylight is best; don’t film into the sun.": ["Лучше снимать днём и не против солнца.", "Kunduzi suratga oling, quyoshga qarshi emas."],
  "VIN photo: close, sharp, no glare.": ["Фото VIN: близко, чётко, без бликов.", "VIN surati: yaqindan, aniq, yaltirashsiz."],
  "Plates and faces are blurred automatically.": ["Номера и лица размываются автоматически.", "Raqamlar va yuzlar avtomatik xiralashtiriladi."],
  "Create passport": ["Создать паспорт", "Pasport yaratish"],
  "Select a walkaround video to start.": ["Выберите видео, чтобы начать.", "Boshlash uchun videoni tanlang."],
  "Japan": ["Япония", "Yaponiya"],
  "China": ["Китай", "Xitoy"],
  "Uzbekistan": ["Узбекистан", "O‘zbekiston"],
  "South Korea": ["Южная Корея", "Janubiy Koreya"],
  "Germany": ["Германия", "Germaniya"],
  "United States": ["США", "AQSH"],
  "Choose a walkaround video to start.": ["Выберите видео, чтобы начать.", "Boshlash uchun videoni tanlang."],
  "Uses the demo files on the server. Nothing is uploaded.": ["Используются демо-файлы на сервере. Ничего не загружается.", "Serverdagi demo fayllar ishlatiladi. Hech narsa yuklanmaydi."],
  "Uploads the video (and VIN photo) to the server.": ["Видео (и фото VIN) загружаются на сервер.", "Video (va VIN surati) serverga yuklanadi."],
  "Demo files are already on the server.": ["Демо-файлы уже на сервере.", "Demo fayllar allaqachon serverda."],
  "Upload complete.": ["Загрузка завершена.", "Yuklash tugadi."],
  "Queued on the server…": ["В очереди на сервере…", "Serverda navbatda…"],
  "Analysing the walkaround…": ["Анализ видео…", "Video tahlil qilinmoqda…"],
  "Passport ready.": ["Паспорт готов.", "Pasport tayyor."],
  "The inspection failed.": ["Осмотр не удался.", "Ko‘rik muvaffaqiyatsiz tugadi."],
  "Waiting for the server…": ["Ожидание сервера…", "Server kutilmoqda…"],
  "Start over": ["Начать заново", "Qaytadan boshlash"],
  "Loading inspection": ["Загрузка осмотра", "Ko‘rik yuklanmoqda"],
  "load models": ["загрузка моделей", "modellarni yuklash"],
  "extract frames": ["кадры из видео", "kadrlarni ajratish"],
  "filter frames": ["отбор кадров", "kadrlarni saralash"],
  "anonymize": ["размытие номеров и лиц", "raqam va yuzlarni xiralashtirish"],
  "segment": ["поиск повреждений и деталей", "shikast va detallarni topish"],
  "dedupe": ["объединение находок", "topilmalarni birlashtirish"],
  "cost": ["расчёт стоимости", "narxni hisoblash"],
  "vin": ["чтение VIN", "VIN o‘qish"],
  "hash frames": ["отпечаток кадров", "kadrlar izi"],
  "done": ["готово", "tayyor"],
  // Car tab
  "Photos": ["Фото", "Suratlar"],
  "Minor": ["Лёгкое", "Yengil"],
  "Moderate": ["Среднее", "O‘rtacha"],
  "Severe": ["Сильное", "Jiddiy"],
  "Low confidence": ["Под вопросом", "Noaniq"],
  "Tap a part to see its damage.": ["Нажмите на деталь, чтобы увидеть повреждения.", "Shikastni ko‘rish uchun detalni bosing."],
  "Drag to turn · ← → keys": ["Тяните, чтобы повернуть · ← →", "Burish uchun suring · ← →"],
  "No confirmed damage": ["Подтверждённых повреждений нет", "Tasdiqlangan shikast yo‘q"],
  "Inspection summary": ["Итоги осмотра", "Ko‘rik xulosasi"],
  "damages found": ["повреждений найдено", "ta shikast topildi"],
  "damage found": ["повреждение найдено", "ta shikast topildi"],
  "Automatic: check each one on the car.": ["Автоматическая оценка: проверьте каждое на машине.", "Avtomatik baho: har birini mashinada tekshiring."],
  "Nothing detected. Still check the car.": ["Ничего не найдено. Всё равно осмотрите машину.", "Hech narsa topilmadi. Baribir mashinani tekshiring."],
  "Estimated repair cost": ["Оценка стоимости ремонта", "Ta’mirlash narxi (taxminiy)"],
  "Starting prices": ["Начальные цены", "Boshlang‘ich narxlar"],
  "nothing to repair": ["ремонт не нужен", "ta’mir kerak emas"],
  "no price data": ["нет цены", "narx yo‘q"],
  "View damage list": ["Список повреждений", "Shikastlar ro‘yxati"],
  "Since last inspection": ["С прошлого осмотра", "Oldingi ko‘rikdan beri"],
  "No new damage": ["Новых повреждений нет", "Yangi shikast yo‘q"],
  "First inspection": ["Первый осмотр", "Birinchi ko‘rik"],
  "of this VIN with AvtoPassport": ["этого VIN в AvtoPassport", "bu VIN uchun AvtoPassportda"],
  "Not compared": ["Не сравнивалось", "Solishtirilmagan"],
  "Ready": ["Готово", "Tayyor"],
  "Building…": ["Строится…", "Qurilmoqda…"],
  "this can take several minutes": ["это займёт несколько минут", "bir necha daqiqa oladi"],
  "Not available": ["Недоступно", "Mavjud emas"],
  "on this server": ["на этом сервере", "bu serverda"],
  "Failed": ["Ошибка", "Xato"],
  "point cloud": ["облако точек", "nuqtalar buluti"],
  "high-quality model + point cloud": ["детальная модель + облако точек", "batafsil model + nuqtalar buluti"],
  "PDF ready": ["PDF готов", "PDF tayyor"],
  "No PDF": ["Нет PDF", "PDF yo‘q"],
  "Drag to turn the car. Tap a part to see its repair.": ["Поворачивайте машину пальцем. Нажмите на деталь, чтобы увидеть ремонт.", "Mashinani burish uchun suring. Ta’mirni ko‘rish uchun detalni bosing."],
  "check": ["проверить", "tekshirish"],
  "View in Damage list": ["Открыть в списке повреждений", "Shikastlar ro‘yxatida ko‘rish"],
  "No damage was detected on any part.": ["Ни на одной детали повреждений не найдено.", "Hech bir detalda shikast topilmadi."],
  "Make not decoded": ["Марка не определена", "Marka aniqlanmadi"],
  "✓ Check digit": ["✓ Контрольная цифра", "✓ Nazorat raqami"],
  "Check each one on the car.": ["Проверьте каждое на машине.", "Har birini mashinada tekshiring."],
  "this inspection": ["этот осмотр", "shu ko‘rik"],
  "Owners, accidents, insurance:": ["Владельцы, ДТП, страховка:", "Egalar, avariyalar, sug‘urta:"],
  "registry demo (synthetic data)": ["демо реестра (синтетические данные)", "reyestr demosi (sintetik ma’lumotlar)"],
  "Check digit fails": ["Контрольная цифра не совпадает", "Nazorat raqami mos emas"],
  // parts viewer
  "Take apart": ["Разобрать", "Qismlarga ajratish"],
  "Photo": ["Фото", "Surat"],
  "Parts": ["Детали", "Detallar"],
  "Auto-rotate": ["Автоповорот", "Avtoaylanish"],
  "Reset": ["Сброс", "Qayta tiklash"],
  "Drag to turn · pinch or scroll to zoom · tap a part": ["Тяните для поворота · щипок или колесо для масштаба · нажмите на деталь", "Burish uchun suring · kattalashtirish uchun chimchilang · detalni bosing"],
  "Tap a part in 3D or in the list.": ["Нажмите на деталь в 3D или в списке.", "3D da yoki ro‘yxatda detalni bosing."],
  "Tap a part in 3D or in the list. Sizes need the car's published length, so they are not shown.": ["Нажмите на деталь в 3D или в списке. Для размеров нужна официальная длина машины, поэтому они не показаны.", "3D da yoki ro‘yxatda detalni bosing. O‘lchamlar uchun mashinaning rasmiy uzunligi kerak, shuning uchun ko‘rsatilmaydi."],
  "Front": ["Перед", "Old"],
  "Left side": ["Левая сторона", "Chap tomon"],
  "Right side": ["Правая сторона", "O‘ng tomon"],
  "Rear": ["Зад", "Orqa"],
  "Top": ["Верх", "Tepa"],
  "Points": ["Точек", "Nuqtalar"],
  "Seen in": ["Видно на", "Ko‘rilgan"],
  "Find the part:": ["Найти деталь:", "Detalni topish:"],
  "Size (L × W × H)": ["Размер (Д × Ш × В)", "O‘lcham (U × E × B)"],
  // Damage tab
  "Each damage, what to do about it, and the starting price.": ["Каждое повреждение, что с ним делать и начальная цена.", "Har bir shikast, nima qilish kerakligi va boshlang‘ich narxi."],
  "Damage and repair cost": ["Повреждения и стоимость ремонта", "Shikastlar va ta’mirlash narxi"],
  "Part": ["Деталь", "Detal"],
  "Severity": ["Степень", "Darajasi"],
  "Action": ["Действие", "Harakat"],
  "Cost": ["Цена", "Narx"],
  "dent": ["вмятина", "botiq"],
  "scratch": ["царапина", "tirnalish"],
  "crack": ["трещина", "yoriq"],
  "glass shatter": ["разбитое стекло", "singan oyna"],
  "lamp broken": ["разбитая фара", "singan chiroq"],
  "tire flat": ["спущенная шина", "teshilgan shina"],
  "minor": ["лёгкое", "yengil"],
  "moderate": ["среднее", "o‘rtacha"],
  "severe": ["сильное", "jiddiy"],
  "not rated": ["без оценки", "baholanmagan"],
  "Polish": ["Полировка", "Silliqlash"],
  "Paint": ["Покраска", "Bo‘yash"],
  "Repair": ["Ремонт", "Ta’mirlash"],
  "Replace": ["Замена", "Almashtirish"],
  "none": ["нет", "yo‘q"],
  "estimate": ["оценка", "taxmin"],
  "Often a reflection.": ["Часто это блик.", "Ko‘pincha bu aks."],
  "Estimated repair total": ["Итого за ремонт (оценка)", "Ta’mirlash jami (taxminiy)"],
  "Starting prices from Tashkent shops. Tap a price for its source.": ["Начальные цены мастерских Ташкента. Нажмите на цену, чтобы увидеть источник.", "Toshkent ustaxonalarining boshlang‘ich narxlari. Manbani ko‘rish uchun narxni bosing."],
  "How severity was rated": ["Как оценивалась степень", "Daraja qanday baholandi"],
  "No damage was detected in the walkaround frames.": ["На кадрах повреждений не найдено.", "Kadrlarda shikast topilmadi."],
  // VIN tab
  "The car’s VIN and what changed since its last inspection.": ["VIN машины и что изменилось с прошлого осмотра.", "Mashina VIN raqami va oxirgi ko‘rikdan beri nima o‘zgargani."],
  "Check the VIN against the car’s documents.": ["Сверьте VIN с документами машины.", "VIN raqamini mashina hujjatlari bilan solishtiring."],
  "read": ["прочитан", "o‘qildi"],
  "typed": ["введён", "kiritildi"],
  "Manufacturer": ["Производитель", "Ishlab chiqaruvchi"],
  "Country": ["Страна", "Mamlakat"],
  "Make": ["Марка", "Marka"],
  "Model": ["Модель", "Model"],
  "Model year": ["Модельный год", "Model yili"],
  "Body": ["Кузов", "Kuzov"],
  "Fuel": ["Топливо", "Yoqilg‘i"],
  "Plant country": ["Страна завода", "Zavod mamlakati"],
  "Plant city": ["Город завода", "Zavod shahri"],
  "New since last inspection": ["Новое с прошлого осмотра", "Oldingi ko‘rikdan beri yangi"],
  "No new damage since the last inspection": ["Новых повреждений с прошлого осмотра нет", "Oxirgi ko‘rikdan beri yangi shikast yo‘q"],
  "First inspection of this VIN": ["Первый осмотр этого VIN", "Bu VIN uchun birinchi ko‘rik"],
  "The next inspection will show what changed.": ["Следующий осмотр покажет, что изменилось.", "Keyingi ko‘rik nima o‘zgarganini ko‘rsatadi."],
  "How changes are judged": ["Как оцениваются изменения", "O‘zgarishlar qanday baholanadi"],
  "Open": ["Открыть", "Ochish"],
  // 3D tab
  "Drag to rotate. Pinch or scroll to zoom.": ["Тяните, чтобы вращать. Щипок или колесо — масштаб.", "Aylantirish uchun suring. Kattalashtirish uchun chimchilang."],
  "Parts model": ["Модель по деталям", "Detallar modeli"],
  "new": ["новое", "yangi"],
  "Detailed 3D model": ["Детальная 3D-модель", "Batafsil 3D model"],
  "Quick 3D model": ["Быстрая 3D-модель", "Tezkor 3D model"],
  "Details": ["Подробнее", "Batafsil"],
  "Reset view": ["Сбросить вид", "Ko‘rinishni tiklash"],
  "Turn 90°": ["Повернуть на 90°", "90° burish"],
  "Flip upside down": ["Перевернуть", "Ag‘darish"],
  "Points −": ["Точки −", "Nuqtalar −"],
  "Points +": ["Точки +", "Nuqtalar +"],
  "Hide labels": ["Скрыть подписи", "Yozuvlarni yashirish"],
  "Show labels": ["Показать подписи", "Yozuvlarni ko‘rsatish"],
  "Building the 3D model… (a few minutes)": ["Строится 3D-модель… (несколько минут)", "3D model qurilmoqda… (bir necha daqiqa)"],
  "Waiting to start…": ["Ожидание запуска…", "Boshlanishi kutilmoqda…"],
  // Customize
  "Pick a view of the car, then tap a colour.": ["Выберите вид машины, затем нажмите на цвет.", "Mashina ko‘rinishini tanlang, so‘ng rangni bosing."],
  "Paint try-on": ["Примерка цвета", "Rangni sinab ko‘rish"],
  "Colours": ["Цвета", "Ranglar"],
  "Original": ["Оригинал", "Asl"],
  "How it works": ["Как это работает", "Bu qanday ishlaydi"],
  "White": ["Белый", "Oq"],
  // Paint check
  "Paint check": ["Проверка краски", "Bo‘yoq tekshiruvi"],
  "trade-in": ["трейд-ин", "trade-in"],
  "Trade-in: photograph each gap between two panels. Panels with different paint are marked for a paint-gauge check.": ["Трейд-ин: сфотографируйте каждый зазор между деталями. Детали с другой краской отмечаются для проверки толщиномером.", "Trade-in: detallar orasidagi har bir tirqishni suratga oling. Bo‘yog‘i farq qiladigan detallar qalinlik o‘lchagich bilan tekshirish uchun belgilanadi."],
  "Stand about half a metre away, gap in the middle of the photo. Tap a number or \"Photo\".": ["Встаньте примерно в полуметре, зазор — в центре кадра. Нажмите номер или «Фото».", "Taxminan yarim metr masofada turing, tirqish kadr o‘rtasida bo‘lsin. Raqamni yoki «Surat»ni bosing."],
  "Photos are checked on this phone and never uploaded. It shows which panels to gauge; it does not set the price.": ["Фото проверяются на телефоне и никуда не загружаются. Показывает, какие детали проверить толщиномером, но не определяет цену.", "Suratlar telefonda tekshiriladi va yuklanmaydi. Qaysi detallarni o‘lchash kerakligini ko‘rsatadi, narxni belgilamaydi."],
  "Flagged panels in 3D": ["Отмеченные детали в 3D", "Belgilangan detallar 3D da"],
  "Same paint": ["Краска одинаковая", "Bo‘yoq bir xil"],
  "Small difference": ["Небольшая разница", "Kichik farq"],
  "Different paint": ["Другая краска", "Boshqa bo‘yoq"],
  "Original paint": ["Заводская краска", "Zavod bo‘yog‘i"],
  "Check with a gauge": ["Проверить толщиномером", "O‘lchagich bilan tekshirish"],
  "Gauge first": ["Проверить в первую очередь", "Birinchi navbatda o‘lchash"],
  "Likely repainted": ["Вероятно, перекрашена", "Ehtimol qayta bo‘yalgan"],
  "Gauge these panels:": ["Проверьте толщиномером:", "Ushbu detallarni o‘lchang:"],
  "No panel stands out. Gauge as usual.": ["Ни одна деталь не выделяется. Проверяйте как обычно.", "Hech bir detal ajralib turmaydi. Odatdagidek o‘lchang."],
  "Gap not found": ["Зазор не найден", "Tirqish topilmadi"],
  "Gap too close to the edge": ["Зазор слишком близко к краю", "Tirqish chetga juda yaqin"],
  "Too little paint in view": ["Мало краски в кадре", "Kadrda bo‘yoq kam"],
  "Trim or sticker in the way": ["Мешает накладка или наклейка", "Qoplama yoki stiker xalal beradi"],
  "Too much glare": ["Слишком много бликов", "Juda ko‘p yaltirash"],
  "Too dark": ["Слишком темно", "Juda qorong‘i"],
  "Blurry": ["Размыто", "Xira"],
  "Front left fender": ["Переднее левое крыло", "Old chap qanot"],
  "Front right fender": ["Переднее правое крыло", "Old o‘ng qanot"],
  "Front left door": ["Передняя левая дверь", "Old chap eshik"],
  "Front right door": ["Передняя правая дверь", "Old o‘ng eshik"],
  "Rear left door": ["Задняя левая дверь", "Orqa chap eshik"],
  "Rear right door": ["Задняя правая дверь", "Orqa o‘ng eshik"],
  "Rear left quarter": ["Заднее левое крыло", "Orqa chap qanot"],
  "Rear right quarter": ["Заднее правое крыло", "Orqa o‘ng qanot"],
  "Front left corner, bumper to fender": ["Передний левый угол: бампер — крыло", "Old chap burchak: bamper — qanot"],
  "Hood edge, left side": ["Край капота, слева", "Kapot cheti, chap tomon"],
  "Left side, fender to front door": ["Слева: крыло — передняя дверь", "Chap tomon: qanot — old eshik"],
  "Left side, front door to rear door": ["Слева: передняя — задняя дверь", "Chap tomon: old eshik — orqa eshik"],
  "Left side, rear door to quarter": ["Слева: задняя дверь — заднее крыло", "Chap tomon: orqa eshik — orqa qanot"],
  "Rear left corner, quarter to bumper": ["Задний левый угол: крыло — бампер", "Orqa chap burchak: qanot — bamper"],
  "Trunk edge, left side": ["Край багажника, слева", "Yukxona cheti, chap tomon"],
  "Trunk edge, right side": ["Край багажника, справа", "Yukxona cheti, o‘ng tomon"],
  "Rear right corner, bumper to quarter": ["Задний правый угол: бампер — крыло", "Orqa o‘ng burchak: bamper — qanot"],
  "Right side, quarter to rear door": ["Справа: заднее крыло — задняя дверь", "O‘ng tomon: orqa qanot — orqa eshik"],
  "Right side, rear door to front door": ["Справа: задняя — передняя дверь", "O‘ng tomon: orqa eshik — old eshik"],
  "Right side, front door to fender": ["Справа: передняя дверь — крыло", "O‘ng tomon: old eshik — qanot"],
  "Hood edge, right side": ["Край капота, справа", "Kapot cheti, o‘ng tomon"],
  "Front right corner, fender to bumper": ["Передний правый угол: крыло — бампер", "Old o‘ng burchak: qanot — bamper"],
  "Loading…": ["Загрузка…", "Yuklanmoqda…"],
  // Shops
  "Body and paint shops near you (OpenStreetMap).": ["Кузовные и покрасочные мастерские рядом (OpenStreetMap).", "Yaqin atrofdagi kuzov va bo‘yash ustaxonalari (OpenStreetMap)."],
  "Nearby body shops": ["Мастерские рядом", "Yaqin ustaxonalar"],
  "Use my location": ["Моё местоположение", "Mening joylashuvim"],
  "Use Tashkent centre": ["Центр Ташкента", "Toshkent markazi"],
  "Map": ["Карта", "Xarita"],
  "Website": ["Сайт", "Veb-sayt"],
  // Report
  "Download the PDF. Its fingerprint (SHA-256) proves the photos were not changed later.": ["Скачайте PDF. Его отпечаток (SHA-256) доказывает, что фото не меняли позже.", "PDF faylni yuklab oling. Uning izi (SHA-256) suratlar keyin o‘zgartirilmaganini isbotlaydi."],
  "Report and verification": ["Отчёт и проверка", "Hisobot va tekshiruv"],
  "Download PDF passport": ["Скачать PDF-паспорт", "PDF pasportni yuklab olish"],
  "Download JSON report": ["Скачать JSON-отчёт", "JSON hisobotni yuklab olish"],
  "Fingerprint (SHA-256)": ["Отпечаток (SHA-256)", "Iz (SHA-256)"],
  "Created": ["Создан", "Yaratilgan"],
  "Inspection": ["Осмотр", "Ko‘rik"],
  "Frames analysed": ["Проанализировано кадров", "Tahlil qilingan kadrlar"],
  "Privacy: plates and faces are blurred before anything is stored.": ["Конфиденциальность: номера и лица размываются до сохранения.", "Maxfiylik: raqamlar va yuzlar saqlashdan oldin xiralashtiriladi."],
  "Privacy: plates and faces are blurred before anything is stored or shown.": ["Конфиденциальность: номера и лица размываются до сохранения и показа.", "Maxfiylik: raqamlar va yuzlar saqlash va ko‘rsatishdan oldin xiralashtiriladi."],
  "Processing details": ["Детали обработки", "Ishlov berish tafsilotlari"],
};

// Car part names. Russian adjectives agree with the noun's gender (m / f / n).
const PART = {
  "front bumper": ["передний бампер", "m", "old bamper"],
  "back bumper": ["задний бампер", "m", "orqa bamper"],
  "rear bumper": ["задний бампер", "m", "orqa bamper"],
  "grille": ["решётка радиатора", "f", "radiator panjarasi"],
  "front grille": ["решётка радиатора", "f", "radiator panjarasi"],
  "hood": ["капот", "m", "kapot"],
  "roof": ["крыша", "f", "tom"],
  "trunk": ["крышка багажника", "f", "yukxona qopqog‘i"],
  "trunk lid": ["крышка багажника", "f", "yukxona qopqog‘i"],
  "windshield": ["лобовое стекло", "n", "old oyna"],
  "back windshield": ["заднее стекло", "n", "orqa oyna"],
  "rear windshield": ["заднее стекло", "n", "orqa oyna"],
  "license plate": ["номерной знак", "m", "davlat raqami"],
  "back door": ["задняя дверь", "f", "orqa eshik"],
  "rear door": ["задняя дверь", "f", "orqa eshik"],
  "front door": ["передняя дверь", "f", "old eshik"],
  "back wheel": ["заднее колесо", "n", "orqa g‘ildirak"],
  "rear wheel": ["заднее колесо", "n", "orqa g‘ildirak"],
  "front wheel": ["переднее колесо", "n", "old g‘ildirak"],
  "back window": ["стекло задней двери", "n", "orqa eshik oynasi"],
  "rear door glass": ["стекло задней двери", "n", "orqa eshik oynasi"],
  "front window": ["стекло передней двери", "n", "old eshik oynasi"],
  "front door glass": ["стекло передней двери", "n", "old eshik oynasi"],
  "fender": ["переднее крыло", "n", "old qanot"],
  "front fender": ["переднее крыло", "n", "old qanot"],
  "quarter panel": ["заднее крыло", "n", "orqa qanot"],
  "rear quarter panel": ["заднее крыло", "n", "orqa qanot"],
  "rocker panel": ["порог", "m", "ostona"],
  "headlight": ["фара", "f", "fara"],
  "tail light": ["задний фонарь", "m", "orqa chiroq"],
  "mirror": ["боковое зеркало", "n", "yon oyna"],
  "side mirror": ["боковое зеркало", "n", "yon oyna"],
  "unknown": ["неизвестная деталь", "f", "noma’lum detal"],
};
const SIDE_RU = { left: { m: "левый", f: "левая", n: "левое" }, right: { m: "правый", f: "правая", n: "правое" } };
const SIDE_UZ = { left: "chap", right: "o‘ng" };
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

// "Left back door" / "left_back-door" / "Rear bumper" -> the name in lang (null if unknown)
export function partName(text, lang) {
  const t = String(text || "").replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim().toLowerCase();
  const m = t.match(/^(left|right) (.+)$/);
  const side = m ? m[1] : null;
  const base = m ? m[2] : t;
  const p = PART[base];
  if (!p) return null;
  if (lang === "en") return cap(t);
  if (lang === "ru") return cap(side ? `${SIDE_RU[side][p[1]]} ${p[0]}` : p[0]);
  return cap(side ? `${SIDE_UZ[side]} ${p[2]}` : p[2]);
}

const IDX = { ru: 0, uz: 1 };
const word = (en, lang) => (E[en] ? E[en][IDX[lang]] : en);

// Patterns for text with numbers or names in it: [regex, (match, lang) => text]
const RULES = [
  [/^(.+) · condition passport$/, (m, l) => `${m[1]} · ${l === "ru" ? "паспорт состояния" : "holat pasporti"}`],
  [/^Inspection (\S+) · (.+)$/, (m, l) => `${l === "ru" ? "Осмотр" : "Ko‘rik"} ${m[1]} · ${m[2]}`],
  [/^Server ready(.*)$/, (m, l) => `${l === "ru" ? "Сервер готов" : "Server tayyor"}${m[1]}`],
  [/^Server up(.*)$/, (m, l) => `${l === "ru" ? "Сервер работает" : "Server ishlayapti"}${m[1]}`],
  [/^(.+) · already on the server, no upload$/, (m, l) => `${m[1]} · ${l === "ru" ? "уже на сервере, без загрузки" : "serverda bor, yuklash shart emas"}`],
  [/^Demo: (.+)$/, (m, l) => `${l === "ru" ? "Демо" : "Demo"}: ${m[1]}`],
  [/^(\d+) damages? found$/, (m, l) => (l === "ru" ? `Найдено повреждений: ${m[1]}` : `${m[1]} ta shikast topildi`)],
  [/^\+(\d+) uncertain\. (.+)$/, (m, l) => `+${m[1]} ${l === "ru" ? "под вопросом." : "ta noaniq."} ${word(m[2], l)}`],
  [/^from ([\d\s  ]+) UZS$/, (m, l) => (l === "ru" ? `от ${m[1]} сум` : `${m[1]} so‘mdan`)],
  [/^with uncertain items: from ([\d\s  ]+) UZS$/, (m, l) => (l === "ru" ? `с сомнительными: от ${m[1]} сум` : `noaniqlar bilan: ${m[1]} so‘mdan`)],
  [/^Starting prices · (\d+) items? without a price$/, (m, l) => (l === "ru" ? `Начальные цены · без цены: ${m[1]}` : `Boshlang‘ich narxlar · narxsiz: ${m[1]}`)],
  [/^(\d+) items? without a price$/, (m, l) => (l === "ru" ? `Без цены: ${m[1]}` : `Narxsiz: ${m[1]}`)],
  [/^compared with (.+)$/, (m, l) => (l === "ru" ? `сравнение с ${m[1]}` : `${m[1]} bilan solishtirildi`)],
  [/^Compared with the inspection of (.+)\.$/, (m, l) => (l === "ru" ? `Сравнение с осмотром от ${m[1]}.` : `${m[1]} dagi ko‘rik bilan solishtirildi.`)],
  [/^(\d+) new damage$/, (m, l) => (l === "ru" ? `Новых повреждений: ${m[1]}` : `${m[1]} ta yangi shikast`)],
  [/^(.+) · VIN (read|typed|ambiguous|could not read)$/, (m, l) => `${m[1]} · VIN ${word(m[2], l)}`],
  [/^VIN (read|typed|ambiguous|could not read)$/, (m, l) => `VIN ${word(m[1], l)}`],
  [/^frames SHA-256 (.+)$/, (m, l) => (l === "ru" ? `SHA-256 кадров ${m[1]}` : `kadrlar SHA-256 ${m[1]}`)],
  [/^Parts with findings · (\d+)$/, (m, l) => (l === "ru" ? `Детали с повреждениями · ${m[1]}` : `Shikastlangan detallar · ${m[1]}`)],
  [/^View (\d+) of (\d+)(?: · (\d+) damage box(?:es)?)?$/, (m, l) => (l === "ru" ? `Вид ${m[1]} из ${m[2]}${m[3] ? ` · рамок: ${m[3]}` : ""}` : `${m[2]} tadan ${m[1]}-ko‘rinish${m[3] ? ` · ${m[3]} ta ramka` : ""}`)],
  [/^View (\d+) · (\d+) box(?:es)?$/, (m, l) => (l === "ru" ? `Вид ${m[1]} · рамок: ${m[2]}` : `${m[1]}-ko‘rinish · ${m[2]} ta ramka`)],
  [/^Original · view (\d+) of (\d+)$/, (m, l) => (l === "ru" ? `Оригинал · вид ${m[1]} из ${m[2]}` : `Asl · ${m[2]} tadan ${m[1]}-ko‘rinish`)],
  [/^(.+) · view (\d+) of (\d+)$/, (m, l) => (l === "ru" ? `${m[1]} · вид ${m[2]} из ${m[3]}` : `${m[1]} · ${m[3]} tadan ${m[2]}-ko‘rinish`)],
  [/^Model year (.+?)(?: · (.+))?$/, (m, l) => `${l === "ru" ? "Модельный год" : "Model yili"} ${m[1].replace(" or ", l === "ru" ? " или " : " yoki ")}${m[2] ? ` · ${word(m[2], l)}` : ""}`],
  [/^one of (.+)$/, (m, l) => (l === "ru" ? `один из: ${m[1]}` : `quyidagilardan biri: ${m[1]}`)],
  [/^(\d+) parts · (\d+) with damage$/, (m, l) => (l === "ru" ? `Деталей: ${m[1]} · с повреждениями: ${m[2]}` : `${m[1]} ta detal · ${m[2]} tasi shikastlangan`)],
  [/^(\d+) parts$/, (m, l) => (l === "ru" ? `Деталей: ${m[1]}` : `${m[1]} ta detal`)],
  [/^(\d+) parts from (\d+) anonymised frames, COLMAP poses\. Slide "Take apart" to split the car\.$/, (m, l) => (l === "ru" ? `${m[1]} деталей по ${m[2]} размытым кадрам. Сдвиньте «Разобрать», чтобы разделить машину.` : `${m[2]} ta xiralashtirilgan kadrdan ${m[1]} ta detal. Mashinani ajratish uchun «Qismlarga ajratish»ni suring.`)],
  [/^(\d+)\+ views$/, (m, l) => (l === "ru" ? `${m[1]}+ видах` : `${m[1]}+ ko‘rinishda`)],
  [/^(\d+) views? · score ([\d.]+)$/, (m, l) => (l === "ru" ? `видов: ${m[1]} · оценка ${m[2]}` : `${m[1]} ko‘rinish · ball ${m[2]}`)],
  [/^(\d+) \+ (\d+) to verify$/, (m, l) => (l === "ru" ? `${m[1]} + ${m[2]} проверить` : `${m[1]} + ${m[2]} tekshirish`)],
  [/^Uncertain - please check \((\d+)\)$/, (m, l) => (l === "ru" ? `Под вопросом — проверьте (${m[1]})` : `Noaniq — tekshiring (${m[1]})`)],
  [/^Unchanged \((\d+)\)$/, (m, l) => (l === "ru" ? `Без изменений (${m[1]})` : `O‘zgarmagan (${m[1]})`)],
  [/^Decode notes \((\d+)\)$/, (m, l) => (l === "ru" ? `Примечания расшифровки (${m[1]})` : `Izohlar (${m[1]})`)],
  [/^All AvtoPassport inspections of this VIN \((\d+)\)$/, (m, l) => (l === "ru" ? `Все осмотры этого VIN (${m[1]})` : `Bu VIN ning barcha ko‘riklari (${m[1]})`)],
  [/^(\d+) findings?$/, (m, l) => (l === "ru" ? `находок: ${m[1]}` : `${m[1]} ta topilma`)],
  [/^Built in (.+?) from (\d+) frames\.$/, (m, l) => (l === "ru" ? `Построена за ${m[1]} по ${m[2]} кадрам.` : `${m[2]} ta kadrdan ${m[1]} da qurildi.`)],
  [/^Built in (.+?)\.$/, (m, l) => (l === "ru" ? `Построена за ${m[1]}.` : `${m[1]} da qurildi.`)],
  [/^Reused: same video as on (.+)\.$/, (m, l) => (l === "ru" ? `Использована сохранённая модель: то же видео, что ${m[1]}.` : `Saqlangan model ishlatildi: ${m[1]} dagi bilan bir xil video.`)],
  [/^Inspection history \((\d+)\)$/, (m, l) => (l === "ru" ? `История осмотров (${m[1]})` : `Ko‘riklar tarixi (${m[1]})`)],
  [/^Show all \((\d+)\)$/, (m, l) => (l === "ru" ? `Показать все (${m[1]})` : `Hammasini ko‘rsatish (${m[1]})`)],
  [/^Same video as the inspection of (.+), so the saved model was reused(.*)\.$/, (m, l) => (l === "ru" ? `То же видео, что в осмотре от ${m[1]}, поэтому использована сохранённая модель${m[2]}.` : `${m[1]} dagi ko‘rik bilan bir xil video, shuning uchun saqlangan model ishlatildi${m[2]}.`)],
  [/^(\d+) places? within (\d+) km of (.+)\.$/, (m, l) => (l === "ru" ? `Мест в радиусе ${m[2]} км: ${m[1]}.` : `${m[2]} km radiusda ${m[1]} ta joy.`)],
  [/^Searching within (\d+) km of (.+)…$/, (m, l) => (l === "ru" ? `Поиск в радиусе ${m[1]} км…` : `${m[1]} km radiusda qidirilmoqda…`)],
  [/^No shops found within (\d+) km of (.+)\.$/, (m, l) => (l === "ru" ? `В радиусе ${m[1]} км мастерских не найдено.` : `${m[1]} km radiusda ustaxona topilmadi.`)],
  [/^(\d+) of (\d+) gaps measured$/, (m, l) => (l === "ru" ? `Измерено зазоров: ${m[1]} из ${m[2]}` : `${m[2]} tadan ${m[1]} ta tirqish o‘lchandi`)],
  [/^Retake: (.+)$/, (m, l) => `${l === "ru" ? "Переснимите" : "Qayta oling"}: ${word(m[1], l)}`],
  [/^Uploading… (.+)$/, (m, l) => (l === "ru" ? `Загрузка… ${m[1]}` : `Yuklanmoqda… ${m[1]}`)],
  [/^(dent|scratch|crack|glass shatter|lamp broken|tire flat) \((minor|moderate|severe|not rated)\)$/, (m, l) => `${word(m[1], l)} (${word(m[2], l)})`],
];

export function translate(text, lang = getLang()) {
  if (lang === "en" || !text) return null;
  const s = text.replace(/\u00a0/g, " ").replace(/\s+/g, " ").trim();
  if (!s) return null;
  if (E[s]) return E[s][IDX[lang]];
  const pn = partName(s, lang);
  if (pn && /^[A-Za-z][a-z -]+$/.test(s)) return pn;
  for (const [re, fn] of RULES) {
    const m = s.match(re);
    if (m) return fn(m, lang);
  }
  return null;
}

const ATTRS = ["placeholder", "aria-label", "title", "data-label"];

function translateAttrs(node, lang) {
  for (const a of ATTRS) {
    const v = node.getAttribute(a);
    if (v) {
      const tr = translate(v, lang);
      if (tr != null && tr !== v) node.setAttribute(a, tr);
    }
  }
}

function translateNode(node, lang) {
  if (node.nodeType === Node.TEXT_NODE) {
    const tr = translate(node.data, lang);
    if (tr != null && tr !== node.data.trim()) {
      const lead = node.data.match(/^\s*/)[0];
      const trail = node.data.match(/\s*$/)[0];
      node.data = lead + tr + trail;
    }
    return;
  }
  if (node.nodeType !== Node.ELEMENT_NODE || node.closest("script, style, code, .vin-value, .hash")) return;
  translateAttrs(node, lang);
  for (const child of node.childNodes) translateNode(child, lang);
}

// Translate the page now and every text the app adds or changes later.
export function startTranslator(lang = getLang()) {
  document.documentElement.lang = lang === "uz" ? "uz-Latn" : lang;
  if (lang === "en") return;
  translateNode(document.body, lang);
  new MutationObserver((muts) => {
    for (const m of muts) {
      if (m.type === "characterData") translateNode(m.target, lang);
      else if (m.type === "attributes") translateAttrs(m.target, lang);
      else for (const n of m.addedNodes) translateNode(n, lang);
    }
  }).observe(document.body, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ATTRS });
}

// Search links that find this part for this car: OLX.uz (listings are mostly in Russian) and Google (page language).
export function partSearch(partKey, car, fallbackName = "") {
  const lang = getLang();
  const ru = partName(partKey, "ru") || fallbackName;
  const name = partName(partKey, lang) || fallbackName;
  const extra = { en: "car part", ru: "запчасть купить", uz: "ehtiyot qism" }[lang];
  const olx = [car, ru.toLowerCase()].filter(Boolean).join(" ").trim();
  const g = [car, name, extra].filter(Boolean).join(" ").trim();
  return {
    olx: `https://www.olx.uz/list/q-${encodeURIComponent(olx.replace(/\s+/g, "-"))}/`,
    google: `https://www.google.com/search?q=${encodeURIComponent(g)}`,
    query: olx,
  };
}
