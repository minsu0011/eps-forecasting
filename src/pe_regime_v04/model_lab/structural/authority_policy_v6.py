"""Generated V6 pre-audit authority; formal spent execution remains disabled."""

AUTHORITY_POLICY = {
    "format_version": 4,
    "mode": "structural_v4_two_layer_external_audit_pinned_authority",
    "activation_relative_path": (
        "outputs/model_zoo_structural_wave_spent_screen_v6_20260819/ACTIVATION_CANDIDATE_V6.json"
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
            "raw_sha256": "b443eb4f6a490f0807ef88f5118a5be10d0fda811596747df176b221ee7c6a55",
            "logical_sha256": "dae3569367375e7aca53b9df73aa7af33258a4beb982034e533c3ccad77e8817",
        },
    },
    "source_inventory_sha256": "5028c8504e7113174cf19e2073267b80be2cda8af1dc7a2088718f815ba64cda",
    "formal_spent_activated": False,
    "activation_authority": {
        "state": "NO_GO_PENDING_INDEPENDENT_V6_AUDIT",
        "v5_independent_no_go_audit_raw_sha256": (
            "53d64d5ff92388478d5ad409fef8c76e1ac13ce976c66adc1d22f4979913a9c0"
        ),
        "v5_independent_no_go_audit_logical_sha256": (
            "179ba0f778dadf280ffdb3d8a8c00dfa5455de6274fba06e8f040415bc08af73"
        ),
        "physical_prediction_runner_v6_raw_sha256": (
            "a392554d6b980d6e006d082913b0f6faddb7dd16bc4f035d89e6838820723f34"
        ),
        "authorization_v6_raw_sha256": (
            "ea8369a6338725eaaca8ce4e23b8ac9083a1a3261b3b967c83ac9cf703956307"
        ),
        "runtime_environment_v6_raw_sha256": (
            "fd41ccf93204f9b8248c681f09fa676b23cc4eb2149acebb7ff3ff369e832e00"
        ),
        "test_evidence_v6_raw_sha256": (
            "30f8a12fb00763c49c6e62ed29302f8bd580661dfeed6776db40fd849dfab1bc"
        ),
        "complete_package_count": 38,
        "lightgbm": "4.6.0",
        "pyyaml": "6.0.2",
        "worker_failure_identity_and_shutdown_receipt": True,
    },
    "manifest_sha256": "f513ce0bb3dceb9a3db52fadcd6d82d4cdcc93d135df3a957e30aa1c2564bd92",
}
