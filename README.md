# passgan-test
<br>Программа для моделирования пула атак с помощью нейронных сетей (PassGAN и LLama3.1)
<br>Взлом пароля, получение саммари по переписке и генерация фишинговых сообщений на его основе.
<br><i>Для работы необходимо локально развернуть LLama и PassGAN.</i>
<br>
<br><b>db.py</b> - база данных с захешированными паролями и сообщениями между пользователями
<br>
<br><b>models.db</b> - базовые классы пользователей и их сообщений
<br>
<br><b>server.py</b> - основная логика. Методы для подключения пользователей, их регистрации (отправка пароля, хэширование и сохранение в бд, получение JWT-токена) и отправки сообщений. Подбор пароля осуществляется через скрипт в cmd. Для упрощения моделирования использован метод get_messages (подключение с именем пользователя и подобранным паролям для получения всеё переписки пользователя). Там же находятся методы для подключения к LLaMA для получения саммари и генерации фишинговых сообщений.
