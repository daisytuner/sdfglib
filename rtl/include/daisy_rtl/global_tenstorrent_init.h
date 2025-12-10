#pragma once

#include <memory>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include <unordered_map>

#ifndef DAISY_TT_DEVICE_IDX
#define DAISY_TT_DEVICE_IDX 0
#endif

namespace daisy::tenstorrent {

struct __daisy_DeviceDeleter {
    explicit __daisy_DeviceDeleter(int id = -1) : id(id) {}

    void operator()(tt::tt_metal::IDevice* ptr) const {
        // Device mgmt of Tenstorrent is itself a global var that gets descructed at C++ exit handlers. So it may have
        // already destructed the devices...
    }

private:
    int id;
};

__attribute__((weak)) std::unordered_map<int, std::unique_ptr<tt::tt_metal::IDevice, __daisy_DeviceDeleter>>
    __daisy_global_tt_devices;

__attribute__((weak)) tt::tt_metal::IDevice* daisy_get_tt_device(int device_id = 0) {
    auto& holder = __daisy_global_tt_devices[device_id];
    auto* ptr = holder.get();
    if (!ptr) {
        if (tt::DevicePool::is_initialized()) {
            auto& pool = tt::DevicePool::instance();
            if (pool.is_device_active(device_id)) {
                ptr = pool.get_active_device(device_id);
            }
        }
        if (!ptr) {
            ptr = tt::tt_metal::CreateDevice(DAISY_TT_DEVICE_IDX);
        }
        holder = std::unique_ptr<tt::tt_metal::IDevice, __daisy_DeviceDeleter>(ptr, __daisy_DeviceDeleter(device_id));
    }
    return ptr;
}

__attribute__((weak)) bool daisy_force_close_tt_device(tt::tt_metal::IDevice* device) {
    auto id = device->id();
    auto& holder = __daisy_global_tt_devices[id];
    auto* ptr = holder.get();
    holder.reset();
    return ptr != nullptr;
}

__attribute__((weak)) void daisy_force_close_all_tt_devices() { __daisy_global_tt_devices.clear(); }

} // namespace daisy::tenstorrent
