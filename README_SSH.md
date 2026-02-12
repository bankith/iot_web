# SSH Connection Guide to KIMQ 6490

This guide explains how to connect to an KIMQ 6490 via SSH from Windows, macOS, and Linux.

## Prerequisites
Before you begin, ensure you have:
1.  **IP Address**: The IP address of the target KIMQ 6490 (e.g., `192.168.1.50`).
2.  **Username**: The username on the remote KIMQ 6490. (default user : root)
3.  **Password**: The user's password OR an SSH key pair for authentication. (default password : Green_7650!)
4.  **Network**: Both machines must be on the same network (or the target must be accessible via the internet).

## 1. From Windows

Windows 10/11 comes with a built-in OpenSSH client.

1.  Open **PowerShell** or **Command Prompt** (search "cmd" or "powershell" in the Start menu).
2.  Run the ssh command:
    ```powershell
    ssh root@ip_address
    ```
    *Example:* `ssh root@192.168.1.100`
3.  If this is your first time connecting, type `yes` to accept the fingerprint.
4.  Enter your password when prompted.

## 2. From macOS and Linux

Both macOS and Linux have the SSH client installed by default.

1.  Open the **Terminal** app.
2.  Run the connection command:
    ```bash
    ssh root@ip_address
    ```
    *Example:* `ssh root@192.168.0.100`
3.  Type `yes` if prompted to verify the host authenticity.
4.  Enter your password.
