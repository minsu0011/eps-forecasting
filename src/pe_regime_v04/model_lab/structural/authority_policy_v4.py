"""Generated V4 spent authority; its exact bytes require an external audit pin."""

AUTHORITY_POLICY = {
    "format_version": 4,
    "mode": "structural_v4_two_layer_external_audit_pinned_authority",
    "activation_relative_path": (
        "outputs/model_zoo_structural_wave_spent_screen_20260819/ACTIVATION_BINDING_EXECUTION.json"
    ),
    "exact_bindings": {
        "trigger_decision": {
            "raw_sha256": "45b6dee894ed72809c49902f4f78ff058a465db6ebfe8903a21c659883504765",
            "logical_sha256": "a3440ae5a3aa81767738d59e4f60ae391e0034d6e223c6f8513089042c1b7023",
        },
        "base_binding_lock_v4": {
            "raw_sha256": "b60ca510b4cad411c29dded212df934e327b948c24f0a932060e1bdff26a81a4",
            "logical_sha256": "23f10ffdbbf4f81157997baf54a770d47d764eb7196bed4bee6b945b2293bf47",
        },
        "execution_snapshot_v4": {
            "raw_sha256": "b1996943814339ca3b9b9aa4b1d92c13bfe569783e9a174b9440bff54705dec2",
            "logical_sha256": "f42c7569fd362fddc843d0b81dc34323a819db99f591662a405f651818befd0c",
        },
        "independent_audit_activation": {
            "raw_sha256": "ed469d17308afbaaec23558ee31230912c9daca9936b765046f335b74b4da467",
            "logical_sha256": "8d3d314ffa73b86cc5543e7eb636cd7050625511cc5aa4403aefcdf9f0fe280a",
        },
    },
    "source_inventory_sha256": ("5028c8504e7113174cf19e2073267b80be2cda8af1dc7a2088718f815ba64cda"),
    "formal_spent_activated": True,
    "activation_authority": {
        "independent_go_audit_raw_sha256": (
            "d59656aeea97a568f52fb17c95d114a4a28a0ac15d23208fe98aa9e90dfba5c9"
        ),
        "independent_go_audit_logical_sha256": (
            "efe869aa58ac89b7c5c5a82f86ff8533f2d66f551ef1f9620abb05664ad9102f"
        ),
        "v4_reaudit_request_raw_sha256": (
            "93d0985e2f6ff479577ebfc235caee886a7b3deafee9b57e95de48d73918ff6b"
        ),
        "audited_pre_activation_policy_raw_sha256": (
            "ca874545c461d60cdfd71569472c743943e0f74db5080aeb8e1443e6d41aef2e"
        ),
        "v4_implementation_manifest_raw_sha256": (
            "9be2956cf283ae51fd117181fa7d13d604e09c1863823a671482d411b1ddb4e5"
        ),
        "predict_inputs_raw_sha256": (
            "f8f30ab74d812909b644984ba9ee3c4a8f509dfabb2ab566c7048313245d500c"
        ),
        "formal_schedule_sha256": (
            "1d2423311c01397dafca26f7a865be7fb470fd337897e714a8a975a32f559c00"
        ),
        "formal_coverage_sha256": (
            "2cbc336c9ea85a7cbc42af253e6f259940ee748189cdf4f4cc38c66545866f37"
        ),
        "physical_prediction_runner_raw_sha256": (
            "cc5c3345baf6319a2e1f5612ba9735b0cbd86943810a9219c3c9f87502e398f0"
        ),
        "predecessor_activation_raw_sha256": (
            "4166124875e1ec074ca0d6facdaf63c7a940327a23574de7ca39190e9e2ef750"
        ),
        "predecessor_external_policy_pin_raw_sha256": (
            "9e190c46aa10af9f948e13046646d3f24bccce2f7f08ac08e4e2ae290b18fd52"
        ),
    },
    "manifest_sha256": "e7af154af059038d210f779a4e1c0d94de5dff87bc4de78a505b7b0064030511",
}
