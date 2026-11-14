class SuspiciousActivity:
    def check(self, state, intrusion):
        suspicious = False
        reason = ""

        if intrusion == "INTRUSION":
            suspicious = True
            reason = "INTRUSION"

        elif state == "LOITERING":
            suspicious = True
            reason = "LOITERING"

        elif state == "RAPID MOVEMENT":
            suspicious = True
            reason = "RAPID MOVEMENT"

        return suspicious, reason
