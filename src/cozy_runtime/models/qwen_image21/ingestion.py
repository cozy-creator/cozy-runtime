"""Reviewed Qwen source configuration and native checkpoint preparation."""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

REPOSITORY = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
FILES: dict[str, tuple[int, str]] = {
    "transformer/config.json": (
        370,
        "56ae3281c4e6c2d1aa3658252d187488071815fd79bef15808bb0205fc1a2241",
    ),
    "text_encoder/config.json": (
        1517,
        "6001331949f7f86ab6d791a80b12246d4f8da5159fa08c3e62368a39f36d63a9",
    ),
    "vae/config.json": (2079, "9785d527b278cb8b210e9f48a8028d92a4190968ce7e24c8fc85d9d1b82f6ba2"),
    "scheduler/scheduler_config.json": (
        485,
        "5895f3a167c14a967fe9ac70c64924ae5acc79799e0679fd12907e594a713cd1",
    ),
    "processor/tokenizer_config.json": (
        5445,
        "81ec7bb9530159b326c0bef1d0b6c33d392090524014ea3f0123a3c1eb9c2af5",
    ),
    "processor/preprocessor_config.json": (
        782,
        "93585062a80db5e8ca038efc7726a3e6411d9db948472d81d63c6303993be8c5",
    ),
    "processor/video_preprocessor_config.json": (
        817,
        "59c5c9eb52182eb14c06ffb10ca9effd29adce5f238a95de23ca14a38dbd2cb1",
    ),
    "processor/chat_template.jinja": (
        5292,
        "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4",
    ),
    "LICENSE": (7831, "8dc973f024ff95966bea25866efa443fd16776dcb1001e681e3d467ea572b28d"),
    "processor/tokenizer.json": (
        11422654,
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
}

# Exact constructor census at Diffusers4295ee3, Transformers5.17; no tensor values.
_CENSUS = (
    "eNq1nWFTGzkahP9LPlNUujUzHv-WrS0XSZwsF8AETPbu35_tQALGNoz0dN2xtdzhF71StwT0Y81ff31Y313c3H9d3V0v7z6c"
    "fVhfXi8X6-V_14vl9afll_Pt5_fr5e2vT78s786vLm-WF3cLnf-7vPz2z_rD2V_dx_lw5n74--zDp68aPvx9Vl3Ve1W3_zhR"
    "dlPw8uZ8V_hm8z--fPGbr7u8WVxd_G_zzSu-5-phXfHiy-tv2xe_fMnQHX_B9erLw9XF-nJ1c_5swjWUsXtrnH8-W3y6Wn3-"
    "fn_-8fxivd5M12rxY1rLp0p950r95EptlmfzGVNuq6znMyaP1WW-t5TZquf66vb89m71n-d1PI41zT2V287Vy5nalays9u1i"
    "vdx3RuUQxelVnF7F6VWsXsXoVYxexepVqF6V0Ks5vZrTqzm9mtWrGb2a0atZvRrVqxN6LZxeC6fXwum1sHotjF4Lo9fC6rWg"
    "ei0JvXacXjtOrx2n147Va8fotWP02rF67VC9dgm99pxee06vPafXntVrz-i1Z_Tas3rtUb32Cb0OnF4HTq8Dp9eB1evA6HVg"
    "9Dqweh1QvQ4Jvc44vc44vc44vc5Yvc4Yvc4Yvc5Yvc5Qvc4Seh05vY6cXkdOryOr15HR68jodWT1OqJ6HRN6nXN6nXN6nXN6"
    "nbN6nTN6nTN6nbN6naN6nUfyAjDgEphwCYy4BGdcgkIuQSmX4JhLbM6lTNBFJl1k1EVmXXTYRaVdVNxF511w4BVJvARGXgIz"
    "L4Ghl-DUS1DsJSj3Ehx8iU2-FIm-BGZfAsMvgemX4PhLUP4lKAATnICJjcAUycAEhmACUzCBMZjgHExQECYoCRMchYnNwhQJ"
    "wwSmYQLjMIF5mOBATFAiJigSE5yJiQ3FFEnFBMZiAnMxgcGY4GRMUDQmKBsTHI6JTccUiccE5mMCAzKBCZngiExQRiYoJBOc"
    "komNyRTJyQQGZQKTMoFRmeCsTFBYJigtExyXic3LFAnMBCZmAiMzgZmZ4NBMUGomKDYTnJuJDc4USc4MJmcGkzODyZnh5MxQ"
    "cmYoOTOcnJlNzhxJzgwmZwaTM4PJmeHkzFByZig5M5ycmU3OnHmvGPlmMfLdYuTbxej3i1FvGKPeMUa_ZQx-z1gkOTOYnBlM"
    "zgwmZ4aTM0PJmaHkzHByZjY5cyQ5M5icGUzODCZnhpMzQ8mZoeTMcHJmNjlzJDkzmJwZTM4MJmeGkzNDyZmh5MxwcmY2OXMk"
    "OTOYnBlMzgwmZ4aTM0PJmaHkzHByZjY5cyQ5M5icGUzODCZnhpMzQ8mZoeTMcHJmNjlzJDkzmJwZTM4MJmeGkzNDyZmh5Mxw"
    "cmY2OXMkOTOYnBlMzgwmZ4aTM0PJmaHkzHByZjY5cyQ5K2ByVsDkrIDJWYGTswIlZwVKzgqcnBU2OSuR5KyAyVkBk7MCJmcF"
    "Ts4KlJwVKDkrcHJW2OSstCdnuxnajuXX_cCTFnA7JYsXfQwHLs3dXUp883n15em-3eXV-c_L-4eLq_Pbi_Xnfx5vLN6bX_U-"
    "K2c-02Ze6ut9ury4f6z27hqr-8cKvwfj8rE7m1Lj952t29nVy64qa1R08qKCgVG4aRQ7Q_74_mzPKV0_nLXVehzQtlJdkQOy"
    "qx9RreZ-V9k6-fGi7q-fn18AXq3AvYqPg9vWayu1p6czomL9zAlwmpqdJsBpanaaQKeJcJpIpwlxmnCniXOacKeJc5oBp7nZ"
    "aQac5manGXSaCaeZdJoRpxl3mjmnGXeaOacVwGml2WkFcFppdloBnVYIpxXSaQVxWsGdVjinFdxphXNaBzita3ZaBzita3Za"
    "BzqtI5zWkU7rEKd1uNM6zmkd7rSOc1oPOK1vdloPOK1vdloPOq0nnNaTTusRp_W403rOaT3utJ5z2gA4bWh22gA4bWh22gA6"
    "bSCcNpBOGxCnDbjTBs5pA-60gXPaDHDarNlpM8Bps2anzUCnzQinzUinzRCnzXCnzTinzXCnzTinjYDTxmanjYDTxmanjaDT"
    "RsJpI-m0EXHaiDtt5Jw24k4bOafNAafNm502B5w2b3baHHTanHDanHTaHHHaHHfanHPaHHfaHEyuCUhE7ZSICExE7ZyISFBE"
    "CCkiFBURw4qIh0UE0iLicRGRvAgCjADECIKMAMwICo0w1AiLjUDcSAAcIcmRADoCsiMi4BG10yMi8BG18yMiARIhBIlQhEQM"
    "QyIeIhFIkYjHSARyJCJAErWTJCJQErWzJCJhEiE0iVCcRAxPIh4oEUiUiEdKBDIlIqAStVMlIrAStXMlIsESIWSJULREDFsi"
    "Hi4RSJeIx0sE8iUiABO1EyYiEBO1MyYiIRMhlIlQzEQMZyIeNBFImohHTQSyJiJgE7XTJiJwE7XzJiKBEyHEiVDkRAxzIh46"
    "EUidiMdOBHInIsATtZMnItATtbMnIuETIfSJUPxEDH8iHkARSKCIR1AEMigiIBS1UygiMBS1cygiQRQhJIpQFEUMiyIeRhFI"
    "o4jHUQTyKCKAFLUTKSKQFLUzKSKhFCFUilAsRQyXIh5MEUimiEdTBLIpJtgUt7MpJtgUt7MpJtkUI2yKUTbFDJtink0xyKaY"
    "Z1MMsikm2BS3sykm2BS3sykm2RQjbIpRNsUMm2KeTTHIpphnU0zea4JcbALcbIJcbQLcbYJebsLcbsJebwLdbxK44IS84SRw"
    "xQnIpphgU9zOpphgU9zOpphkU4ywKUbZFDNsink2xSCbYp5NMcimmGBT3M6mmGBT3M6mmGRTjLApRtkUM2yKeTbFIJtink0x"
    "yKaYYFPczqaYYFPczqaYZFOMsClG2RQzbIp5NsUgm2KeTTHIpphgU9zOpphgU9zOpphkU4ywKUbZFDNsink2xSCbYp5NMcGm"
    "XC_vvi3vdiqtU_rzAvXf_-CyDR_Hs-0_6ms9rVdDEe_fsN9U62lAE-7c_7Jc3t6vLz5_X_wqt7i6vF8_XvD-crKQihVTdqwe"
    "sqjvKJ4Ycsuyv6M4pgPhOhCsAyV1oIgOlNSBIjowrgPDOnBSB47owEkduFYHVxc33x4uvi0Xvz7dPftksV59X97cPzvCe83L"
    "cFZTcPdQmu3THu6XV18Xv35eXLz8We_ws2bqan_fq62P7qjaP4O1V8E5-bHY-4HMIzLTXNXfjzB612OWplZ-uM3U_bL69-bQ"
    "sr1-ktOUypc3tw_rX89y2tuEW8Z7u7pfb9dtebO-XN3A5RU0t4LmVtDcCppbEXMrYm7FzK2QuRUztyLmVtbcDprbQXM7aG4H"
    "ze2IuR0xt2PmdsjcjpnbEXM7a-4SNHcJmrsEzV2C5i4Rc5eIuUvM3CVk7hIzd4mYu2TN3QXN3QXN3QXN3QXN3UXM3UXM3cXM"
    "3YXM3cXM3UXM3WXN3QfN3QfN3QfN3QfN3UfM3UfM3cfM3YfM3cfM3UfM3WfNPQTNPQTNPQTNPQTNPUTMPUTMPcTMPYTMPcTM"
    "PUTMPWTNPQuaexY09yxo7lnQ3LOIuWcRc89i5p6FzD2LmXsWMfcsa-4xaO4xaO4xaO4xaO4xYu4xYu4xZu4xZO4xZu4xYu4x"
    "a-550NzzoLnnQXPPg-aeR8w9j5h7HjP3PGTueczc84i552GIJYmoKcmoKQmpKUmpKYOpKcOpKQeqKUWqKYeqKcOqKQ2rRWm1"
    "KK4W5dWiwFqIWAsha0FmLQatBam1ELYW5taUBNeUJNeURNeUZNeUgdeUodeUw9eU4teUA9iUIdgURtiUZNiUhNiUpNiUxNiU"
    "4diUAdmUI9mUQtmUY9mUgdkUptmUxNmU5NmUBNqUJNqUQdqUYdqUg9qUotqUw9qU4doUBtuUJNuURNuUZNuUhNuUoduUwduU"
    "49uUAtyUI9yUQdwUZtyUhNyUpNyUxNyU5NyUAd2UId2UQ92UYt2Ug92Uod0Uxt2U5N2UBN6UJN6URN6UYd6Ugd6Uo96Uwt6U"
    "496UAd8UJt-URN-UZN-UhN-UpN-Uwd-U4d-UA-CUIuCUQ-CUYeAUhuCUpOCUxOCU5OCUBOGUIeGUQeGUY-GUguGUo-GUweEU"
    "5uGc5OGc5OGc5OGc5OGc4eGc4eGc4-Gc4uGc4-Gc4eEc5uGc5OGc5OGc5OGc5OGc4eGc4eGc4-Gc4uGc4-Gc4eGcvsctepFb"
    "9Ca36FVu0bvcQpe5hW5zC17nFrvPLXihW-hGtzAP5yQP5yQP5yQP5yQP5wwP5wwP5xwP5xQP5xwP5wwP5zAP5yQP5yQP5yQP"
    "5yQP5wwP5wwP5xwP5xQP5xwP5wwP5zAP5yQP5yQP5yQP5yQP5wwP5wwP5xwP5xQP5xwP5wwP5zAP5yQP5yQP5yQP5yQP5wwP"
    "5wwP5xwP5xQP5xwP5wwP5zAP5yQP5yQP5yQP5yQP5wwP5wwP5xwP5xQP5xwP5wwP5zAP5yQP5yQP5yQP5yQP5wwP5wwP5xwP"
    "5xQP5xwP5wwP5zAP5yQP5yQP5yQP5yQP5wwP5wwP5xwP5xQP5xwP5wwP5zAPV5I8XEnycCXJw5UkD1cyPFzJ8HAlx8OVFA9X"
    "cjxcyfBwJczDlSQPV5I8XEnycCXJw5UMD1cyPFzJ8XAlxcOVHA9XMjxcCfNwJcnDlSQPV5I8XEnycCXDw5UMD1dyPFxJ8XAl"
    "x8OVDA9X0o84jT7jNPqQ0-hTTqOPOQ095zT0oNPgk05jjzoNPus09LDTMA9XkjxcSfJwJcnDlSQPVzI8XMnwcCXHw5UUD1dy"
    "PFzJ8HAlzMOVJA9XkjxcSfJwJcnDlQwPVzI8XMnxcCXFw5UcD1cyPFxJ8XCTKlxdL_5ZXnx5thK95uW12H9eLDdf_fi688-r"
    "m5-Ly5s_r9ra46yclc1LvhafeMWny4v73dcf_crdEn66Wn3-fn_-8fxueX-zXG__bduWNqK8vr749Q21_c_kMtuB6MXAN_89"
    "NfLThdr6MdOPqX5c3Y-Y9RG1PmpcHzHrI2p9VLk-28_uL65vrzb_z6bW7t_OG-b3SL1pwxJjax23teae0peO-XpTp6qjP8LZ"
    "jqS-Jb9saftR25Nbetpt3_f_rO7Wnx_Wr6a5trs_FauHtr_v1E63DimobrrVKiFBEhInIdVK6K39Z_qQTu9AdSNbX14vF9sO"
    "Xw_s_dN-uODEcfno1jhNBT6-N5axmzTlPrY5bgpV9fRH2buhVPfklz3tPip7cktPRzbHp3mube_V7jh9bPu7Y-1865CG6uZb"
    "rRoSpCFxGlKtht7aHacP6fTuWDeyA7vj08DeP-0nd8f3jqsc3R2nqaAc3x1nwzhpysux3XFTqKqnP8reDqW-J7_saddXZU9u"
    "6enI7vg0z7Xtvdodp49tf3esnW8d0lDdfKtVQ4I0JE5DqtXQW7vj9CGd3h3rRnZgd3wa2Pun_eTu-N5xdUd3x2kq6E7vjlOm"
    "vGvdHTtod-y43bFr3R07aAfquB2oa92BOmgH6rgdqKvZga4vv_x6_fnvv8s_SW-_o6k11qvFj-_PdgmXj91ZW63HhraVplZ4"
    "GZK8Z8c6VWjSzNbvUodqVCj_eJmGPgz0YaYP1_UhYD3ErIda1kPAeohZjwk7z3bMi9Xm59n3D3v3c_DqxR9k_fYof7_q6U9T"
    "fj2oHw8XN-v9v4htau8-Dozo2defqLrLOw-VHrqz4fCvAvsveaw-vN7vviyPZJNS7235Q3Oy_6KnwW9ecvRr33NE7L7noX7e"
    "KLJ_RpSuH84aiz39jr0pNbXCXki_G0flYJ4fE--c3qN_CX0cwrsGcTwi-t3NKWG8eVRU9WKiF1O9uLIXEesial3UtC4i1kXU"
    "umjCujzcvk1LvKObw2Wq1uZUqbaOzHRkrqP6NRKzRuLWSI1rJGaNxK1Rq4_MrJG5NXLjGplZI3Nr5Mo1erg9mStPHs_BevWD"
    "OvBnwd0v_G_-_PRWuVO_87-uIeZMEHcmqPFMEHMmiDsT1HgmiDkTxJ0JajwTxJwJ4s4ENZ4JYs4EcWeCGs8EMWeCuDNBjWeC"
    "4DNBxJkg9kwQcCaYORNOIEb9bJgy0UcJo02din7-KHk3jsp2_LKd7UddN67v5kh2_nt66zp7FZ1PHdj-MVU3zTqkmpppVpto"
    "hIhGlGjUJhoja2Nqbdy2NkbWxtTauG5tTh9LUwdz6lCaPqJD6Ot2a3mc6epq007J42DXlBU_gXVt3172_kk-CnX9ehfZ1G7-"
    "6Hc7jNpu_LKb7UddN67v5siB9DS7dY29Oo-mjmv_PKqbZR3STM0sq00zQjQjSjNq04yRtTG1Nm5bGyNrY2ptXLc2p8-jqYM5"
    "dR69b0THwbUp83sCW1PXTWjpKLS2KVPRzbNfrLuuuhu_7Gb7UdeN67s59i6wx9mta-z1m8AmjuvVH82qZlmHNFMzy2rTjBDN"
    "iNKM2jRjZG1MrY3b1sbI2phaG79_bV7BRW8P_TVc9PYo99Gi3wP6-_8whDaV"
)


def configuration(root: Any) -> dict[str, Any]:
    members: dict[str, bytes] = {}
    for name, (length, digest) in FILES.items():
        with (root / name).open("rb") as stream:
            raw = stream.read(length + 1)
        if len(raw) != length or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Unreviewed Qwen metadata bytes: {name}")
        members[name] = raw
    tokenizer = json.loads(members["processor/tokenizer_config.json"])
    for key in (
        "tokenizer_file",
        "name_or_path",
        "added_tokens_decoder",
        "tokenizer_class",
        "processor_class",
    ):
        tokenizer.pop(key, None)
    result = {
        component: json.loads(members[f"{component}/config.json"])
        for component in ("transformer", "text_encoder", "vae")
    }
    result["scheduler"] = json.loads(members["scheduler/scheduler_config.json"])
    result["processor"] = {
        "tokenizer": base64.urlsafe_b64encode(
            zlib.compress(members["processor/tokenizer.json"], 9)
        ).decode(),
        "tokenizer_config": tokenizer,
        "image_processor": json.loads(members["processor/preprocessor_config.json"]),
        "video_processor": json.loads(members["processor/video_preprocessor_config.json"]),
        "chat_template": members["processor/chat_template.jinja"].decode(),
    }
    return result


def preparation(
    inspection: Any, root: Path
) -> tuple[dict[str, Any], tuple[tuple[str, str], ...], dict[str, bytes]]:
    """Validate a source census and prepare immutable configuration/file values."""
    census = json.loads(zlib.decompress(base64.urlsafe_b64decode(_CENSUS)))
    expected = {(c, k): (tuple(shape), dtype) for c, k, shape, dtype in census}
    actual = {(c, k): t for c, rows in inspection.components.items() for k, t in rows.items()}
    if inspection.configs or set(actual) != set(expected):
        raise ValueError("Preparation requires the exact original source tensor census")
    for pair, (shape, dtype) in expected.items():
        tensor = actual[pair]
        part = tensor.parts.get("value")
        if (
            tuple(tensor.shape) != shape
            or tensor.logical_dtype != dtype
            or tensor.encoding
            != "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f"
            or len(tensor.parts) != 1
            or part is None
            or part.dtype != dtype
            or tuple(part.shape) != shape
        ):
            raise ValueError(f"Source differs from the reviewed native checkpoint: {pair}")
    config = configuration(root)
    provenance: dict[str, str] = {
        "source_repository": REPOSITORY,
        "source_revision": REVISION,
        "conversion": (
            "Native TensorFS layout and processor configuration; original tensor values unchanged."
        ),
        "notice": (
            "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, "
            "Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. "
            "All Rights Reserved."
        ),
        "license_sha256": FILES["LICENSE"][1],
    }
    config["provenance"] = provenance
    return (
        config,
        tuple((c, k) for c, k, _shape, _dtype in census),
        {
            "LICENSE": (root / "LICENSE").read_bytes(),
            "Notice": (provenance["notice"] + "\n").encode(),
            "CHANGES.txt": (provenance["conversion"] + "\n").encode(),
        },
    )
