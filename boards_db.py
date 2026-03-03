# ── MCU Database with detailed specs ──────────────────────────

MCU_BOARDS = [
    {
        'name': 'ATtiny85',
        'family': 'Microchip ATtiny',
        'ram_kb': 8,
        'ram_range': '8 KB',
        'tier': 'Ultra Low-End',
        'price_usd': '$1-2',
        'features': 'Minimal power, small form',
        'use_case': 'Basic IoT sensors'
    },
    {
        'name': 'STM32L0 (L011)',
        'family': 'STMicroelectronics',
        'ram_kb': 16,
        'ram_range': '16-32 KB',
        'tier': 'Ultra Low-End',
        'price_usd': '$2-4',
        'features': 'Ultra-low power, ARM Cortex-M0+',
        'use_case': 'Battery-powered wearables'
    },
    {
        'name': 'STM32L1 (L152)',
        'family': 'STMicroelectronics',
        'ram_kb': 32,
        'ram_range': '32-64 KB',
        'tier': 'Low-End',
        'price_usd': '$3-5',
        'features': 'EEPROM, low power, ARM Cortex-M3',
        'use_case': 'Industrial sensors, smart meters'
    },
    {
        'name': 'nRF51822',
        'family': 'Nordic Semiconductor',
        'ram_kb': 32,
        'ram_range': '32-64 KB',
        'tier': 'Low-End',
        'price_usd': '$5-8',
        'features': 'Bluetooth LE, ARM Cortex-M0',
        'use_case': 'BLE wearables, health trackers'
    },
    {
        'name': 'STM32L4 (L476)',
        'family': 'STMicroelectronics',
        'ram_kb': 64,
        'ram_range': '64-128 KB',
        'tier': 'Mid-Range',
        'price_usd': '$4-7',
        'features': 'High speed, ARM Cortex-M4, FPU',
        'use_case': 'Audio processing, complex algorithms'
    },
    {
        'name': 'nRF52832',
        'family': 'Nordic Semiconductor',
        'ram_kb': 64,
        'ram_range': '64-128 KB',
        'tier': 'Mid-Range',
        'price_usd': '$6-10',
        'features': 'Bluetooth LE + ANT, ARM Cortex-M4, FPU',
        'use_case': 'Advanced BLE applications, IoT hubs'
    },
    {
        'name': 'STM32F4 (F446)',
        'family': 'STMicroelectronics',
        'ram_kb': 128,
        'ram_range': '128-256 KB',
        'tier': 'Upper Mid',
        'price_usd': '$5-9',
        'features': 'High performance, ARM Cortex-M4, FPU, DSP',
        'use_case': 'Image processing, motor control'
    },
    {
        'name': 'ESP32',
        'family': 'Espressif',
        'ram_kb': 160,
        'ram_range': '128-256 KB',
        'tier': 'Upper Mid',
        'price_usd': '$3-7',
        'features': 'WiFi + BLE, dual-core, rich ecosystem',
        'use_case': 'Connected IoT, web-enabled devices'
    },
    {
        'name': 'STM32H7 (H743)',
        'family': 'STMicroelectronics',
        'ram_kb': 256,
        'ram_range': '256+ KB',
        'tier': 'High-End',
        'price_usd': '$8-15',
        'features': 'Ultra-high performance, ARM Cortex-M7, FPU, cache',
        'use_case': 'Real-time control, complex DSP'
    },
    {
        'name': 'ESP32-S3',
        'family': 'Espressif',
        'ram_kb': 320,
        'ram_range': '256+ KB',
        'tier': 'High-End',
        'price_usd': '$4-8',
        'features': 'WiFi 6, BLE 5.2, AI accelerator, dual-core',
        'use_case': 'Advanced ML inference, multimedia'
    }
]


def get_eligible_boards(min_ram_kb, max_ram_kb=None):
    """Get all eligible boards that meet minimum RAM requirement.
    
    Args:
        min_ram_kb: Minimum RAM required by the model
        max_ram_kb: Optional maximum RAM (for cost optimization)
    
    Returns:
        List of eligible board dictionaries sorted by RAM efficiency
    """
    eligible = []
    for board in MCU_BOARDS:
        if board['ram_kb'] >= min_ram_kb:
            if max_ram_kb is None or board['ram_kb'] <= max_ram_kb:
                eligible.append(board)
    
    # Sort by RAM size (smallest capable first for cost efficiency)
    return sorted(eligible, key=lambda x: x['ram_kb'])


def get_board_tier_summary(min_ram_kb, max_ram_kb=None):
    """Get categorized boards by tier for better visualization.
    
    Returns:
        Dictionary with tiers as keys and list of boards as values
    """
    eligible_boards = get_eligible_boards(min_ram_kb, max_ram_kb)
    tiers_dict = {}
    
    for board in eligible_boards:
        tier = board['tier']
        if tier not in tiers_dict:
            tiers_dict[tier] = []
        tiers_dict[tier].append(board)
    
    return tiers_dict
