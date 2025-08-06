import imaplib
import email
import json

def test_email_check(config_path="config/email_config.json"):
    with open(config_path) as f:
        cfg = json.load(f)

    sender = cfg["sender"]
    receiver = cfg["receiver"]
    password = cfg["password"]
    imap_server = cfg.get("imap", "imap.gmail.com")

    try:
        print(f"🔐 Logging into {sender}...")
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(sender, password)
        mail.select("inbox")

        status, data = mail.uid("search", f'(UNSEEN FROM "{receiver}")')
        if status != "OK":
            print("❌ Failed to search inbox.")
            return

        email_ids = data[0].split()
        print(f"📥 Found {len(email_ids)} unseen emails from {receiver}.")

        for uid in email_ids:
            status, msg_data = mail.uid("fetch", uid, "(RFC822)")
            if status != "OK":
                continue
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)
            subject = msg["Subject"]
            print(f"📧 Email detected with subject: {subject}")

        mail.logout()
    except Exception as e:
        print(f"❌ Error during email check: {e}")

if __name__ == "__main__":
    test_email_check()
