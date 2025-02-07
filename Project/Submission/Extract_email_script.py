import mailbox
import os
from bs4 import BeautifulSoup

def save_emails_as_text(input_file, output_folder, top_n=100):
    """
    Extracts the top N emails from an mbox file and saves them as individual text files, removing HTML.

    Parameters:
    - input_file: Path to the source mbox file.
    - output_folder: Folder to save the text files.
    - top_n: Number of emails to extract (default is 100).
    """
    try:
        print(f"Starting to process the mbox file: {input_file}")
        
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output folder created: {output_folder}")
        
        # Open the source mbox file
        print("Opening the mbox file...")
        source_mbox = mailbox.mbox(input_file)
        total_emails = len(source_mbox)
        print(f"Total emails in the mbox file: {total_emails}")
        
        # Initialize a counter for progress
        email_count = 0
        
        for i, message in enumerate(source_mbox):
            if i >= top_n:
                break
            
            print(f"Processing email {i + 1}...")
            subject = message['subject'] or "No Subject"
            from_email = message['from'] or "Unknown Sender"
            print(f"  Subject: {subject}")
            print(f"  From: {from_email}")
            
            # Construct file name for each email
            email_file = os.path.join(output_folder, f"email_{i + 1}.txt")
            
            # Extract and clean the email body
            try:
                body = message.get_payload(decode=True)
                if not body:
                    plain_text = "No Body Content"
                else:
                    decoded_body = body.decode(errors="ignore")
                    soup = BeautifulSoup(decoded_body, "html.parser")
                    plain_text = soup.get_text(separator="\n").strip()
                
                # Save the cleaned email to a text file
                with open(email_file, "w", encoding="utf-8") as f:
                    f.write(f"Subject: {subject}\n")
                    f.write(f"From: {from_email}\n")
                    f.write(f"To: {message['to']}\n")
                    f.write(f"Date: {message['date']}\n\n")
                    f.write(plain_text)
                print(f"  Saved email {i + 1} to {email_file}")
            except Exception as save_error:
                print(f"  Failed to save email {i + 1}: {save_error}")
            
            # Increment and display progress
            email_count += 1
            if email_count % 10 == 0 or email_count == top_n:
                print(f"Processed {email_count}/{top_n} emails.")
        
        print(f"Successfully saved top {top_n} emails to {output_folder}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_mbox = "All mail Including Spam and Trash.mbox"
output_folder = "path_to_your_output_folder"
save_emails_as_text(input_mbox, output_folder, top_n=5000)
