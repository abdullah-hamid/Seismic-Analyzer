# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['Seismic Analyzer.py'],
    pathex=[],
    binaries=[],
    datas=[
    ('logos/ucsd-logo-png-transparent.png', 'logos'),
    ('logos/seismic.ico', 'logos'),
    ('ui_files/*.ui', 'ui_files')
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Seismic Analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\Seismic Analyzer\\logos\\seismic.ico'],
)
