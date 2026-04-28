#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/settings/settings.h>
#include <zephyr/logging/log.h>

#include "ble_kws.h"

LOG_MODULE_REGISTER(ble_kws, LOG_LEVEL_INF);

/* ── UUIDs ─────────────────────────────────────────────────────────────── */

#define BT_UUID_KWS_SVC_VAL \
    BT_UUID_128_ENCODE(0xa7f0b5c1, 0x1234, 0x4567, 0x89ab, 0xcdef01234567ULL)

#define BT_UUID_KWS_NOTIF_VAL \
    BT_UUID_128_ENCODE(0xa7f0b5c2, 0x1234, 0x4567, 0x89ab, 0xcdef01234567ULL)

static struct bt_uuid_128 kws_svc_uuid   = BT_UUID_INIT_128(BT_UUID_KWS_SVC_VAL);
static struct bt_uuid_128 kws_notif_uuid = BT_UUID_INIT_128(BT_UUID_KWS_NOTIF_VAL);

/* ── GATT service ──────────────────────────────────────────────────────── */
/* attrs: [0] primary svc  [1] chrc decl  [2] chrc value  [3] CCC */

static bool notify_enabled;

static void kws_ccc_changed(const struct bt_gatt_attr *attr, uint16_t value)
{
    notify_enabled = (value == BT_GATT_CCC_NOTIFY);
    LOG_INF("KWS notify %s", notify_enabled ? "enabled" : "disabled");
}

BT_GATT_SERVICE_DEFINE(kws_svc,
    BT_GATT_PRIMARY_SERVICE(&kws_svc_uuid),
    BT_GATT_CHARACTERISTIC(&kws_notif_uuid,
        BT_GATT_CHRC_NOTIFY,
        BT_GATT_PERM_NONE, NULL, NULL, NULL),
    BT_GATT_CCC(kws_ccc_changed, BT_GATT_PERM_READ | BT_GATT_PERM_WRITE),
);

/* ── Advertising ───────────────────────────────────────────────────────── */

static const struct bt_data ad[] = {
    BT_DATA_BYTES(BT_DATA_FLAGS, BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR),
    BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_KWS_SVC_VAL),
};

static const struct bt_data sd[] = {
    BT_DATA(BT_DATA_NAME_COMPLETE,
            CONFIG_BT_DEVICE_NAME,
            sizeof(CONFIG_BT_DEVICE_NAME) - 1),
};

static void start_adv(void)
{
    int rc = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
    if (rc) {
        LOG_ERR("adv start failed: %d", rc);
    }
}

/* ── Connection callbacks ──────────────────────────────────────────────── */

static struct bt_conn *current_conn;

static void on_connected(struct bt_conn *conn, uint8_t err)
{
    if (err) {
        LOG_ERR("BLE connect failed: %u", err);
        start_adv();
        return;
    }
    current_conn = bt_conn_ref(conn);
    LOG_INF("BLE connected");
}

static void on_disconnected(struct bt_conn *conn, uint8_t reason)
{
    bt_conn_unref(current_conn);
    current_conn  = NULL;
    notify_enabled = false;
    LOG_INF("BLE disconnected (reason=%u), restarting advertising", reason);
    start_adv();
}

BT_CONN_CB_DEFINE(conn_cbs) = {
    .connected    = on_connected,
    .disconnected = on_disconnected,
};

/* ── Public API ────────────────────────────────────────────────────────── */

int ble_kws_init(void)
{
    int rc = bt_enable(NULL);
    if (rc) {
        LOG_ERR("bt_enable failed: %d", rc);
        return rc;
    }

    /* Required when CONFIG_BT_SETTINGS/CONFIG_SETTINGS is enabled (pulled in
     * by BT_LBS_SECURITY_ENABLED default). Without this bt_le_adv_start fails
     * because the BLE identity address has not been loaded from flash yet. */
    if (IS_ENABLED(CONFIG_SETTINGS)) {
        settings_load();
    }

    start_adv();
    LOG_INF("BLE KWS ready, advertising as \"%s\"", CONFIG_BT_DEVICE_NAME);
    return 0;
}

void ble_kws_notify(keyword_t kw, float conf)
{
    if (!current_conn || !notify_enabled) {
        return;
    }
    uint8_t buf[2] = {
        (uint8_t)kw,
        (uint8_t)(conf * 100.0f + 0.5f),  /* 0-100 percent */
    };
    /* attrs[2] is the characteristic value attribute (decl is at [1]) */
    int rc = bt_gatt_notify(current_conn, &kws_svc.attrs[2], buf, sizeof(buf));
    if (rc) {
        LOG_WRN("bt_gatt_notify failed: %d", rc);
    }
}
