#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <typeinfo>
#include <map>
#include <Eigen/Dense>
#include <iostream>

namespace pyb = pybind11;

Eigen::Matrix4f npArray2Mat(const pyb::array_t<float> &np_mat) {
    auto* np_mat_ptr = (float*) np_mat.request().ptr;
    Eigen::Matrix4f mat;
    for (int r=0; r<4;++r){
        for (int c=0; c<4; ++c){
            mat(r, c) = np_mat_ptr[r*4+c];
            // std::cout << np_mat_ptr[r*4+c] << ", ";
        }
        // std::cout << std::endl;
    }
    return mat;
}

pyb::array_t<float> pointcloudVoxelize(
    const pyb::array_t<float>& np_pointclouds,
    float grid_size
){
    auto* pointcloud_ptr = (float*) np_pointclouds.request().ptr;
    int num_points = np_pointclouds.request().shape[0];
    int dims = np_pointclouds.request().shape[1];

    std::map<std::vector<int>, float> voxel_intensity_map;
    std::map<std::vector<int>, int> voxel_density_map;
    for (int i=0; i<num_points;++i){
        float intensity = pointcloud_ptr[i*dims+3];
        // std::cout << pointcloud_ptr[i*dims] << ", " << pointcloud_ptr[i*dims+1] << ", " << pointcloud_ptr[i*dims+2] <<std::endl;
        auto key = std::vector<int>{
            int(pointcloud_ptr[i*dims] / grid_size),
            int(pointcloud_ptr[i*dims+1] / grid_size),
            int(pointcloud_ptr[i*dims+2] / grid_size),
        };
        if (voxel_density_map.find(key) != voxel_density_map.end()) {
            voxel_density_map[key] += 1;
            voxel_intensity_map[key] += intensity;
        } else {
            voxel_density_map[key] = 1;
            voxel_intensity_map[key] = intensity;
        }
    }
    for (auto &voxel:voxel_density_map){
        voxel_intensity_map[voxel.first] /= static_cast<float>(voxel.second);
    }

    int num_voxel = voxel_density_map.size();
    int out_dims = 5;
    auto result = pyb::array_t<float>(std::vector<pyb::ssize_t>{num_voxel, out_dims});
    auto *result_ptr = (float*)result.request().ptr;
    int count = 0;
    // RETURN x,y,z,i,density
    for (auto&voxel:voxel_intensity_map){
        result_ptr[count*out_dims] = float(voxel.first[0])*grid_size; 
        result_ptr[count*out_dims+1] = float(voxel.first[1])*grid_size;
        result_ptr[count*out_dims+2] = float(voxel.first[2])*grid_size;
        result_ptr[count*out_dims+3] = voxel.second;
        result_ptr[count*out_dims+4] = float(voxel_density_map[voxel.first]);
        count+=1;
    }
    return result;
}

void mappingOccupancyRGB(
    const pyb::array_t<float>& np_query_points,
    const pyb::array_t<float> &np_pt2img,
    const pyb::array_t<float> &np_label,
    const pyb::array_t<float> &np_img_data,
    const pyb::array_t<float> &np_range,
    float grid_size,
    pyb::array_t<float> &np_rgb_results
){
    auto* query_points_ptr = (float*) np_query_points.request().ptr;
    auto* range_ptr = (float*) np_range.request().ptr;
    auto* rgb_results_ptr = (float*) np_rgb_results.request().ptr;
    auto* label_ptr = (float*)np_label.request().ptr;
    int result_dims = np_rgb_results.request().shape[1];
    int num_query_points = np_query_points.request().shape[0];
    int dims = np_query_points.request().shape[1];
    auto* img_data_ptr = (float*) np_img_data.request().ptr;
    int img_h = np_img_data.request().shape[0];
    int img_w = np_img_data.request().shape[1];
    // std::cout << img_h << ", " << img_w << std::endl;

    float min_x = range_ptr[0];
    float max_x = range_ptr[1];
    float min_y = range_ptr[2];
    float max_y = range_ptr[3];
    float min_z = range_ptr[4];
    float max_z = range_ptr[5];

    // 获取候选点
    std::vector<Eigen::Vector4f> valid_points_vec;
    std::vector<int> pt_idx_vec;
    std::vector<Eigen::Vector2f> velocity_vec;
    std::vector<Eigen::Vector3i> bev_idx_vec;
    {
        Eigen::Vector4f pt;
        for (int i=0; i<num_query_points;++i){
            if (label_ptr[i] == 0 || label_ptr[i] == 255) {
                continue;
            }
            pt.x() = query_points_ptr[i*dims];
            pt.y() = query_points_ptr[i*dims+1];
            pt.z() = query_points_ptr[i*dims+2];
            // if (i <=5){
            //     std::cout << pt.x() <<", "<< pt.y() << ", "<< pt.z() <<std::endl;
            // }
            
            pt.w() = 1;
            if (pt.x() < min_x || pt.x()> max_x
                || pt.y() < min_y || pt.y() > max_y
                || pt.z() < min_z || pt.z() > max_z) {
                continue;
            }
            valid_points_vec.emplace_back(pt);
            pt_idx_vec.emplace_back(i);
        }
    }
    auto pt2img_mat = npArray2Mat(np_pt2img);
    std::vector<float> position_shift{
            grid_size/2, grid_size/2, grid_size/2,
            grid_size/2, grid_size/2, -grid_size/2,
            grid_size/2, -grid_size/2, grid_size/2,
            grid_size/2, -grid_size/2, -grid_size/2,

            -grid_size/2, grid_size/2, grid_size/2,
            -grid_size/2, grid_size/2, -grid_size/2,
            -grid_size/2, -grid_size/2, grid_size/2,
            -grid_size/2, -grid_size/2, -grid_size/2,
    };

    // 计算遮挡性
    std::vector<int> visible_map(img_h*img_w, -1);
    std::vector<float> visible_depth_map(img_h*img_w, 99999);
    std::vector<Eigen::Vector2i> img_pos(valid_points_vec.size());
    // std::cout << "compute visibility: " << valid_points_vec.size() << std::endl;
    for (int pt_idx=0; pt_idx<valid_points_vec.size();++pt_idx){
        auto& tmp_pt = valid_points_vec[pt_idx];
        // 将点变换到图片上
        auto pts_2d = tmp_pt.transpose() * pt2img_mat.transpose();
        if (pts_2d.z() <= 0.5){
            continue;
        }
        float map_depth = fmax(1e-6f, fmin(pts_2d.z(), 99999));
        int mapped_w = int(pts_2d.x() / map_depth);
        int mapped_h = int(pts_2d.y() / map_depth);

        if (mapped_h >= 0 && mapped_h < img_h && mapped_w >= 0 && mapped_w < img_w) {
            // std::cout << pt_idx << "," << mapped_w << "," << mapped_h << ", " << map_depth << std::endl;
            img_pos[pt_idx] = Eigen::Vector2i{mapped_h, mapped_w};
            int max_w = 0, max_h = 0;
            int min_w = 99999, min_h = 99999;
            Eigen::Vector4f center_voxel_pt{
                    static_cast<float>(tmp_pt.x()),
                    static_cast<float>(tmp_pt.y()),
                    static_cast<float>(tmp_pt.z()), 1};
            Eigen::Vector4f tmp_voxel_pt;
            for (int i = 0; i < 8; ++i) {
                tmp_voxel_pt.x() = center_voxel_pt.x() + position_shift[i * 3];
                tmp_voxel_pt.y() = center_voxel_pt.y() + position_shift[i * 3 + 1];
                tmp_voxel_pt.z() = center_voxel_pt.z() + position_shift[i * 3 + 2];
                tmp_voxel_pt.w() = 1;
                auto tmp_pts_2d = tmp_voxel_pt.transpose() * pt2img_mat.transpose();
                if (tmp_pts_2d.z() <= 0.5) continue;
                float tmp_map_depth = fmax(1e-6f, fmin(tmp_pts_2d.z(), 99999));
                int tmp_mapped_w = int(tmp_pts_2d.x() / tmp_map_depth);
                int tmp_mapped_h = int(tmp_pts_2d.y() / tmp_map_depth);
                // std::cout << "i: "<< i << tmp_mapped_h << ", " << tmp_mapped_w << " depth: "<< tmp_map_depth;
                if (min_w > tmp_mapped_w) min_w = tmp_mapped_w;
                if (min_h > tmp_mapped_h) min_h = tmp_mapped_h;
                if (max_w < tmp_mapped_w) max_w = tmp_mapped_w;
                if (max_h < tmp_mapped_h) max_h = tmp_mapped_h;
            }
            // std::cout << min_h << "," << max_h << "," << min_w << "," << max_w << std::endl;
            for (int r = min_h; r <= max_h; ++r) {
                if (r < 0 || r >= img_h) continue;
                for (int c = min_w; c <= max_w; ++c) {
                    if (c < 0 || c >= img_w) continue;
                    int tmp_idx = r * img_w + c;
                    if (map_depth < visible_depth_map[tmp_idx]) {
                        visible_map[tmp_idx] = pt_idx;
                        visible_depth_map[tmp_idx] = map_depth;
                    }
                }
            }
        }
    }
    // std::cout <<"set value" << std::endl;
    // 对没有遮挡的像素进行赋值
    for (int i : visible_map){
        if (i>=0) {
            int pt_idx = pt_idx_vec[i];
            // img rgb
            int h = int(img_pos[i].x());
            int w = int(img_pos[i].y());
            int img_idx = h*img_w*3+w*3;
//            std::cout << img_idx << "," << img_data_ptr[img_idx] << std::endl;
            rgb_results_ptr[pt_idx*result_dims] = img_data_ptr[img_idx];
            rgb_results_ptr[pt_idx*result_dims+1] = img_data_ptr[img_idx+1];
            rgb_results_ptr[pt_idx*result_dims+2] = img_data_ptr[img_idx+2];
        }
    }
}

// -------------------------------
PYBIND11_MODULE(occupancy_api, m){
    m.doc() = "cpp operations for python";
    m.def("mappingOccupancyRGB", &mappingOccupancyRGB, "compute rgb value for occupancy");
    m.def("pointcloudVoxelize", &pointcloudVoxelize, "point cloud voxelization");
}
