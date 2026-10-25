"""ISO 14229-1 (UDS) response decoding for the services a read-only assistant needs (stdlib only). It decodes; it never sends a request.

  0x59  ReadDTCInformation positive response (sub-functions 0x01 count, 0x02 DTCs by status mask, 0x0A supported DTCs)
  0x62  ReadDataByIdentifier positive response
  0x7F  negative response (service id + NRC)
DTC numbers follow SAE J2012 / ISO 15031-6 text form (P/C/B/U + 4 hex digits) plus the ISO 14229 failure-type byte."""
import re


class UdsError(ValueError): pass

NRC = {0x10: 'generalReject', 0x11: 'serviceNotSupported', 0x12: 'subFunctionNotSupported', 0x13: 'incorrectMessageLengthOrInvalidFormat', 0x14: 'responseTooLong', 0x21: 'busyRepeatRequest', 0x22: 'conditionsNotCorrect', 0x24: 'requestSequenceError', 0x25: 'noResponseFromSubnetComponent', 0x26: 'failurePreventsExecutionOfRequestedAction', 0x31: 'requestOutOfRange', 0x33: 'securityAccessDenied', 0x34: 'authenticationRequired', 0x35: 'invalidKey', 0x36: 'exceededNumberOfAttempts', 0x37: 'requiredTimeDelayNotExpired', 0x38: 'generalSecurityViolation', 0x39: 'securedModeRequested', 0x3A: 'insufficientProtection', 0x3B: 'terminationWithSignatureRequested', 0x3C: 'accessDenied', 0x3D: 'versionNotSupported', 0x3E: 'securedLinkNotSupported', 0x3F: 'certificateNotAvailable', 0x40: 'auditTrailInformationNotAvailable', 0x50: 'certificateVerificationFailed_InvalidTimePeriod', 0x51: 'certificateVerificationFailed_InvalidSignature', 0x52: 'certificateVerificationFailed_InvalidChainOfTrust', 0x53: 'certificateVerificationFailed_InvalidType', 0x54: 'certificateVerificationFailed_InvalidFormat', 0x55: 'certificateVerificationFailed_InvalidContent', 0x56: 'certificateVerificationFailed_InvalidScope', 0x57: 'certificateVerificationFailed_InvalidCertificate', 0x58: 'ownershipVerificationFailed', 0x59: 'challengeCalculationFailed', 0x5A: 'settingAccessRightsFailed', 0x5B: 'sessionKeyCreationDerivationFailed', 0x5C: 'configurationDataUsageFailed', 0x5D: 'deAuthenticationFailed', 0x70: 'uploadDownloadNotAccepted', 0x71: 'transferDataSuspended', 0x72: 'generalProgrammingFailure', 0x73: 'wrongBlockSequenceCounter', 0x78: 'requestCorrectlyReceived_ResponsePending', 0x7E: 'subFunctionNotSupportedInActiveSession', 0x7F: 'serviceNotSupportedInActiveSession', 0x81: 'rpmTooHigh', 0x82: 'rpmTooLow', 0x83: 'engineIsRunning', 0x84: 'engineIsNotRunning', 0x85: 'engineRunTimeTooLow', 0x86: 'temperatureTooHigh', 0x87: 'temperatureTooLow', 0x88: 'vehicleSpeedTooHigh', 0x89: 'vehicleSpeedTooLow', 0x8A: 'throttlePedalTooHigh', 0x8B: 'throttlePedalTooLow', 0x8C: 'transmissionRangeNotInNeutral', 0x8D: 'transmissionRangeNotInGear', 0x8F: 'brakeSwitchNotClosed', 0x90: 'shifterLeverNotInPark', 0x91: 'torqueConverterClutchLocked', 0x92: 'voltageTooHigh', 0x93: 'voltageTooLow', 0x94: 'resourceTemporarilyNotAvailable'}                                                                         # every negative response code of ISO 14229-1 (names in the standard's lowerCamelCase)
STATUS_BITS = ["testFailed", "testFailedThisOperationCycle", "pendingDTC", "confirmedDTC", "testNotCompletedSinceLastClear", "testFailedSinceLastClear",
               "testNotCompletedThisOperationCycle", "warningIndicatorRequested"]


def dtc_text(b0, b1):
    """2 DTC bytes -> 'P0133' style (bits 15-14: P/C/B/U, 13-12: digit, rest: 3 hex digits)."""
    return "PCBU"[b0 >> 6] + str((b0 >> 4) & 0x3) + format(b0 & 0x0F, "X") + format(b1, "02X")


def status_names(s): return [n for i, n in enumerate(STATUS_BITS) if s >> i & 1]


def decode(payload):
    """UDS message (after ISO-TP) -> dict, or None if it is not a response this module understands."""
    if not payload: return None
    sid = payload[0]
    if sid == 0x7F:
        if len(payload) < 3: return None
        return {"type": "negative", "service": payload[1], "nrc": payload[2], "nrc_name": NRC.get(payload[2], f"0x{payload[2]:02X}")}
    if sid == 0x59 and len(payload) >= 3:
        sub, avail = payload[1], payload[2]; out = {"type": "dtc", "subfunction": sub, "availability_mask": avail, "dtcs": []}
        body = payload[3:]
        if sub == 0x01:
            if len(body) != 3: raise UdsError("invalid DTC count response length")
            out["count"] = int.from_bytes(body[1:3], "big"); return out
        if sub in (0x02, 0x0A, 0x0F, 0x13, 0x15):
            if len(body) % 4: raise UdsError("truncated DTC record")
            for i in range(0, len(body) - 3, 4):
                st = body[i + 3]
                out["dtcs"].append({"code": dtc_text(body[i], body[i + 1]), "failure_type": body[i + 2], "status": st, "status_flags": status_names(st), "raw": int.from_bytes(body[i:i + 3], "big")})
            return out
        raise UdsError(f"unsupported DTC subfunction 0x{sub:02X}")
    if sid == 0x62 and len(payload) >= 3: return {"type": "data", "did": int.from_bytes(payload[1:3], "big"), "data": bytes(payload[3:])}
    return None


def encode_dtc_response(dtcs, sub=0x02, avail=0xFF):
    """Build a positive 0x59 response (tests/simulator). dtcs = [(b0, b1, ftb, status)]"""
    out = bytearray([0x59, sub, avail])
    for b0, b1, ftb, st in dtcs: out += bytes([b0, b1, ftb, st])
    return bytes(out)


def parse_dtc(code):
    """'P1A00' -> (b0, b1) bytes (the inverse of dtc_text)."""
    if not isinstance(code, str) or not re.fullmatch(r"[PCBU][0-3][0-9A-Fa-f]{3}", code): raise ValueError("invalid DTC code")
    code = code.upper()
    sysl = "PCBU".index(code[0]); d = int(code[1]); h = int(code[2], 16); lo = int(code[3:5], 16)
    return (sysl << 6) | (d << 4) | h, lo
