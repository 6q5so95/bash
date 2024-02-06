import os
import stat
import tempfile
import pytest
from pathlib import Path
import win32security
import ntsecuritycon as con

def _set_permissions(path, permissions):
    sd = win32security.GetFileSecurity(path, win32security.DACL_SECURITY_INFORMATION)
    user, _, _ = win32security.LookupAccountName("", win32security.GetUserName())
    acl = win32security.ACL()
    if permissions & stat.S_IRUSR:
        acl.AddAccessAllowedAce(win32security.ACL_REVISION, con.FILE_GENERIC_READ, user)
    else:
        acl.AddAccessDeniedAce(win32security.ACL_REVISION, con.FILE_GENERIC_READ, user)
    sd.SetSecurityDescriptorDacl(1, acl, 0)
    win32security.SetFileSecurity(path, win32security.DACL_SECURITY_INFORMATION, sd)

@pytest.fixture
def setup_files():
    test_file = tempfile.NamedTemporaryFile(delete=False)
    test_file.write(b'This is a test file.')
    test_file.close()
    yield test_file.name
    os.unlink(test_file.name)

def test_no_read_permissions(setup_files):
    file_path = setup_files
    _set_permissions(file_path, 0o200)
    assert _precheck_read_file(file_path) is False
