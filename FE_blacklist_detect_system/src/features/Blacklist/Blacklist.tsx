import { useEffect, useState } from "react";
import { SubmitHandler, useForm } from "react-hook-form";
import * as ProfileBlacklistService from "../../APIServices/ProfileBlacklist.api";
import * as DeleteDataService from "../../APIServices/DeleteData.api";
import * as CreateDataService from "../../APIServices/CreateData.api";
import * as UpdateDataService from "../../APIServices/UpdateData.api";
import * as UpdateModelService from "../../APIServices/UpdateModel.api";

import DataTable from "../../components/DataTable/DataTable";
import { Button, Dialog, DialogTitle, DialogContent, Fab } from "@mui/material";
import { GridColDef } from "@mui/x-data-grid";
import NoteAddIcon from "@mui/icons-material/NoteAdd";
import EditIcon from "@mui/icons-material/Edit";
import Icon from "@mdi/react";
import { mdiTrashCan } from "@mdi/js";
import AddPhotoAlternateIcon from "@mui/icons-material/AddPhotoAlternate";
import AddAPhotoIcon from "@mui/icons-material/AddAPhoto";
import { toast } from "react-toastify";

const Blacklist = () => {
  const [openAddBlacklistData, setOpenAddBlacklistData] = useState(false);
  const [openEditBlacklistData, setOpenEditBlacklistData] = useState(false);
  const [selectedRow, setSelectedRow] = useState<any>(null);
  console.log("selectedRow", selectedRow);

  const [blackList, setBlackList] = useState([]);
  console.log("blackList", blackList);
  const blackListColumns: GridColDef[] = [
    { field: "id", headerName: "ID", width: 50 },
    {
      field: "avatar",
      headerName: "Ảnh nhận dạng",
      renderCell: (params) => (
        <div className="flex justify-center items-center">
          <img className="w-12" src={params.row.avatar} alt="avatar" />,
        </div>
      ),
      width: 150,
    },
    { field: "fullName", headerName: "Họ và tên", width: 200 },
    { field: "yearOfBirth", headerName: "Năm sinh", width: 150 },
    { field: "hometown", headerName: "Quê quán", width: 150 },
    { field: "IdNumber", headerName: "Căn cước/CMND", width: 200 },
    // { field: "violation", headerName: "Vi phạm", width: 200 },
    {
      field: "action",
      headerName: "Hành động",
      renderCell: (params) => (
        <div className="w-full h-full flex justify-evenly items-center">
          <Button
            className="!min-w-10 !w-10 !h-9"
            variant="contained"
            color="primary"
            title="Sửa dữ liệu"
            onClick={() => handleActionEditBlacklistData(params.row)}
          >
            <EditIcon fontSize="small" />
          </Button>

          <Button
            className="!min-w-14 !w-10 !h-9"
            variant="contained"
            color="error"
            title="Xóa dữ liệu"
            onClick={() => handleActionDeleteBlacklistData(params.row)}
          >
            <Icon path={mdiTrashCan} size={1} />
          </Button>
        </div>
      ),
      width: 240,
    },
  ];
  const [paginationModel, setPaginationModel] = useState({
    page: 0,
    pageSize: 5,
  });

  const [imageFileForNewBlacklistedObj, setImageFileForNewBlacklistedObj] =
    useState<File | undefined | null>();
  const [imagesFilesForModel, setImagesFilesForModel] = useState<
    File[] | undefined | null
  >();

  // form variable
  const {
    register,
    handleSubmit,
    formState: { isSubmitting },
    reset,
  } = useForm();

  const handleActionAddBlacklistData = () => {
    setOpenAddBlacklistData(true);
  };

  const handleActionEditBlacklistData = (data) => {
    setSelectedRow(data);
    setOpenEditBlacklistData(true);
  };

  const handleActionDeleteBlacklistData = async (data) => {
    console.log("dataactiondelete:", data);
    const formData = new FormData();
    formData.append("name", data.unsignedFirstName);

    // Log FormData content manually
    for (const [key, value] of formData.entries()) {
      console.log(key, value);
    }

    const response = await DeleteDataService.deleteData({
      data: formData,
      options: { headers: { "Content-Type": "multipart/form-data" } },
    });

    if (response) {
      toast.warning(response.message, {
        onClose: () => setBlackList([]),
      });
    }
  };

  const onChangeHandleImageImport = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (!event.target.files) return;
    const files = Array.from(event.target.files);
    setImageFileForNewBlacklistedObj(files[0]);
  };

  const onChangeHandleMultipleImageImport = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (!event.target.files) return;
    const files = Array.from(event.target.files);
    setImagesFilesForModel(
      files.filter((file) => file.type.startsWith("image/"))
    );
  };

  const handleSubmitActionAddBlacklistData: SubmitHandler<any> = async (
    data
  ) => {
    const formData = new FormData();
    formData.append("avata", imageFileForNewBlacklistedObj as Blob);
    formData.append("name", data.unsignedFirstName);
    formData.append("full_name", data.fullName);
    formData.append("year_of_birth", data.yearOfBirth);
    formData.append("hometown", data.hometown);
    formData.append("id_number", data.IdNumber);
    formData.append("violation", "");
    if (imagesFilesForModel) {
      for (let i = 0; i < imagesFilesForModel.length; i++) {
        console.log(imagesFilesForModel[i]);
        formData.append("data", imagesFilesForModel[i]);
      }
    }
    const response = await CreateDataService.postData({
      data: formData,
      options: { headers: { "Content-Type": "multipart/form-data" } },
    });
    if (response && response.message) {
      return toast.success(response.message, {
        onClose: () => {
          setBlackList([]);
          setImageFileForNewBlacklistedObj(null);
          setImagesFilesForModel(null);
          setOpenAddBlacklistData(false);
        },
      });
    }
  };

  const handleSubmitActionEditBlacklistData: SubmitHandler<any> = async (
    data
  ) => {
    const formData = new FormData();
    formData.append("image", imageFileForNewBlacklistedObj as Blob);
    formData.append("name", data.unsignedFirstName);
    formData.append("full_name", data.fullName);
    formData.append("year_of_birth", data.yearOfBirth);
    formData.append("hometown", data.hometown);
    formData.append("id_number", data.IdNumber);
    formData.append("violation", "");
    if (imagesFilesForModel) {
      for (let i = 0; i < imagesFilesForModel.length; i++) {
        console.log(imagesFilesForModel[i]);
        formData.append("data", imagesFilesForModel[i]);
      }
    }
    const response = await UpdateDataService.updateData({
      data: formData,
      options: { headers: { "Content-Type": "multipart/form-data" } },
    });
    if (response) {
      toast.success(response.message, {
        onClose: () => {
          setBlackList([]);
          setImageFileForNewBlacklistedObj(null);
          setImagesFilesForModel(null);
          setOpenEditBlacklistData(false);
        },
      });
    }
  };

  const handleUpdateModel = async () => {
    const responsePostUpdateModel = await UpdateModelService.postData();
    // console.log("responsePostDeleteData", responsePostUpdateModel);
    if (responsePostUpdateModel) {
      toast.success(responsePostUpdateModel.message, {
        onClose: () => {
          setBlackList([]);
        },
      });
    }
  };

  useEffect(() => {
    function transformBlacklistData(data: Record<string, any>) {
      const rows: any[] = [];
      Object.entries(data).forEach(([key, info], index) => {
        const {
          avata,
          full_name,
          year_of_birth,
          hometown,
          id_number,
          violation,
        } = info;

        rows.push({
          id: index + 1,
          // Optionally prepend the API URL to the avatar:
          avatar: import.meta.env.VITE_API_URL + avata,
          unsignedFirstName: key,
          fullName: full_name,
          yearOfBirth: year_of_birth,
          hometown: hometown,
          IdNumber: id_number,
          violation: violation,
        });
      });
      return rows;
    }

    const getBlacklist = async () => {
      const response = await ProfileBlacklistService.getAllData();

      if (response) {
        const transformedBlacklistData = transformBlacklistData(response);
        console.log("transformedBlacklistData: ", transformedBlacklistData);
        if (transformedBlacklistData) {
          setBlackList(transformedBlacklistData);
        }
      }
    };
    if (!blackList || blackList.length === 0) {
      getBlacklist();
    }
  }, [blackList]);

  useEffect(() => {
    if (openAddBlacklistData) {
      reset({
        unsignedFirstName: "",
        fullName: "",
        yearOfBirth: "",
        hometown: "",
        IdNumber: "",
        violation: "",
      });
    } else if (selectedRow) {
      reset({
        unsignedFirstName: selectedRow.unsignedFirstName,
        fullName: selectedRow.fullName,
        yearOfBirth: selectedRow.yearOfBirth,
        hometown: selectedRow.hometown,
        IdNumber: selectedRow.IdNumber,
        violation: selectedRow.violation,
      });
    }
  }, [openAddBlacklistData, selectedRow, reset]);

  return (
    <div className="relative z-0 w-screen h-[calc(100vh-64px)]">
      <Button
        variant="contained"
        className="!absolute z-1 top-12 right-4 !bg-green-600"
        onClick={handleUpdateModel}
      >
        Cập nhật lại model
      </Button>

      {/* <Button
        className="!absolute z-1 top-28 right-4 !bg-green-600"
        variant="contained"
        color="success"
        title="Thêm dữ liệu mới"
        onClick={handleActionAddBlacklistData}
      >
        <NoteAddIcon fontSize="small" />
      </Button> */}

      <Fab
        color="success"
        style={{
          position: "absolute",
          right: 8,
          top: 105,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          color: "white",
        }}
        title="Thêm đối tượng"
        onClick={handleActionAddBlacklistData}
      >
        <NoteAddIcon />
      </Fab>

      <DataTable
        columns={blackListColumns}
        rows={blackList}
        paginationModel={paginationModel}
        setPaginationModel={setPaginationModel}
      />

      {openAddBlacklistData && (
        <Dialog
          open={openAddBlacklistData}
          onClose={() => setOpenAddBlacklistData(false)}
          fullWidth
          maxWidth="xs"
        >
          <DialogTitle>Thêm dữ liệu</DialogTitle>
          <DialogContent>
            <form
              className="flex flex-col justify-evenly items-center h-[400px]"
              onSubmit={handleSubmit(handleSubmitActionAddBlacklistData)}
            >
              <div className="mb-4 flex justify-evenly items-center w-full">
                <label
                  htmlFor="image-for-blacklist-new-listed-obj"
                  className="text-gray-700 font-medium mb-2 w-[50%]"
                >
                  Ảnh:
                </label>
                <div className="w-full flex justify-evenly items-center">
                  <Button
                    variant="contained"
                    component="label"
                    htmlFor="image-for-blacklist-new-listed-obj"
                  >
                    <AddAPhotoIcon />
                    <input
                      id="image-for-blacklist-new-listed-obj"
                      name="image"
                      accept="image/*"
                      style={{ display: "none" }}
                      type="file"
                      onChange={(event) => onChangeHandleImageImport(event)}
                    />
                  </Button>
                  {imageFileForNewBlacklistedObj && <div>Đã tải lên 1 ảnh</div>}
                </div>
              </div>
              <div className="mb-4 flex justify-evenly items-center w-full">
                <label
                  htmlFor="ten-khong-dau"
                  className="text-gray-700 font-medium mb-2 w-[50%]"
                >
                  Tên không dấu:
                </label>
                <input
                  {...register("unsignedFirstName")}
                  type="text"
                  id="ten-khong-dau"
                  className="w-full border rounded-md py-2 px-3 text-gray-700 leading-tight 
                  focus:outline-none focus:shadow-outline"
                />
              </div>

              <div className="mb-4 flex justify-evenly items-center w-full">
                <label
                  htmlFor="ho-va-ten"
                  className="w-[50%] text-gray-700 font-medium mb-2"
                >
                  Họ và Tên:
                </label>
                <input
                  {...register("fullName")}
                  type="text"
                  id="ho-va-ten"
                  className="w-full border rounded-md py-2 px-3 text-gray-700 leading-tight 
                  focus:outline-none focus:shadow-outline"
                />
              </div>

              <div className="mb-4 flex justify-evenly items-center w-full">
                <label
                  htmlFor="nam-sinh"
                  className="w-[50%] text-gray-700 font-medium mb-2"
                >
                  Năm sinh:
                </label>
                <input
                  {...register("yearOfBirth")}
                  type="text"
                  id="nam-sinh"
                  className="w-full border rounded-md py-2 px-3 text-gray-700 leading-tight 
                  focus:outline-none focus:shadow-outline"
                />
              </div>

              <div className="mb-4 flex justify-evenly items-center w-full">
                <label
                  htmlFor="que-quan"
                  className="w-[50%] text-gray-700 font-medium mb-2"
                >
                  Quê quán:
                </label>
                <input
                  {...register("hometown")}
                  type="text"
                  id="que-quan"
                  className="w-full border rounded-md py-2 px-3 text-gray-700 leading-tight 
                  focus:outline-none focus:shadow-outline"
                />
              </div>

              <div className="mb-4 flex justify-evenly items-center w-full">
                <label
                  htmlFor="so-can-cuoc"
                  className="w-[50%] text-gray-700 font-medium mb-2"
                >
                  Số căn cước:
                </label>
                <input
                  {...register("IdNumber")}
                  type="text"
                  id="so-can-cuoc"
                  className="w-full border rounded-md py-2 px-3 text-gray-700 leading-tight 
                  focus:outline-none focus:shadow-outline"
                />
              </div>

              {/* <div className="mb-4 flex justify-evenly items-center w-full">
                <label
                  htmlFor="vi-pham"
                  className="w-[50%] text-gray-700 font-medium mb-2"
                >
                  Vi phạm:
                </label>
                <input
                  {...register("violation")}
                  type="text"
                  id="vi-pham"
                  className="w-full border rounded-md py-2 px-3 text-gray-700 leading-tight 
                  focus:outline-none focus:shadow-outline"
                />
              </div> */}

              <div className="mb-4 flex justify-evenly items-center w-full">
                <label
                  htmlFor="images-for-model"
                  className="w-[60%] text-gray-700 font-medium mb-2"
                >
                  Thêm dữ liệu model:
                </label>
                <div className="w-full flex justify-evenly items-center">
                  <Button
                    variant="contained"
                    component="label"
                    htmlFor="images-for-model"
                  >
                    <AddPhotoAlternateIcon />
                    <input
                      id="images-for-model"
                      name="image"
                      multiple
                      accept="image/*"
                      style={{ display: "none" }}
                      type="file"
                      onChange={(event) =>
                        onChangeHandleMultipleImageImport(event)
                      }
                    />
                  </Button>
                  {imagesFilesForModel && imagesFilesForModel.length > 0 && (
                    <div>Đã tải lên: {imagesFilesForModel.length} ảnh</div>
                  )}
                </div>
              </div>
              <div className="flex justify-evenly items-center w-full">
                <Button onClick={() => setOpenAddBlacklistData(false)}>
                  Hủy
                </Button>

                <Button disabled={isSubmitting} type="submit">
                  Xác nhận
                </Button>
              </div>
            </form>
          </DialogContent>
        </Dialog>
      )}

      {selectedRow && openEditBlacklistData && (
        <Dialog open={openEditBlacklistData} fullWidth maxWidth="xs">
          <DialogTitle>Sửa dữ liệu</DialogTitle>
          <DialogContent>
            <form
              className="flex flex-col justify-evenly items-center h-[400px]"
              onSubmit={handleSubmit(handleSubmitActionEditBlacklistData)}
            >
              <div className="flex justify-evenly items-center w-full">
                <label
                  htmlFor="image-for-blacklist-new-listed-obj"
                  className="text-gray-700 font-medium mb-2 w-[50%]"
                >
                  Ảnh:
                </label>
                <div className="w-full flex justify-center items-center">
                  <Button
                    variant="contained"
                    component="label"
                    htmlFor="image-for-blacklist-new-listed-obj"
                  >
                    <AddAPhotoIcon />
                    <input
                      id="image-for-blacklist-new-listed-obj"
                      name="image"
                      accept="image/*"
                      style={{ display: "none" }}
                      type="file"
                      onChange={(event) => onChangeHandleImageImport(event)}
                    />
                  </Button>
                </div>
              </div>
              <div className="flex justify-evenly items-center w-full">
                <label
                  htmlFor="ten-khong-dau"
                  className="text-gray-700 font-medium mb-2 w-[50%]"
                >
                  Tên không dấu:
                </label>
                <input
                  {...register("unsignedFirstName")}
                  type="text"
                  id="ten-khong-dau"
                  className="w-full border rounded-md py-2 px-3 text-gray-700 leading-tight 
                  focus:outline-none focus:shadow-outline"
                  defaultValue={selectedRow.unsignedFirstName}
                />
              </div>

              <div className="flex justify-evenly items-center w-full">
                <label
                  htmlFor="ho-va-ten"
                  className="w-[50%] text-gray-700 font-medium mb-2"
                >
                  Họ và Tên:
                </label>
                <input
                  {...register("fullName")}
                  type="text"
                  id="ho-va-ten"
                  className="w-full border rounded-md py-2 px-3 text-gray-700 leading-tight 
                  focus:outline-none focus:shadow-outline"
                  defaultValue={selectedRow.fullName}
                />
              </div>

              <div className="flex justify-evenly items-center w-full">
                <label
                  htmlFor="nam-sinh"
                  className="w-[50%] text-gray-700 font-medium mb-2"
                >
                  Năm sinh:
                </label>
                <input
                  {...register("yearOfBirth")}
                  type="text"
                  id="nam-sinh"
                  className="w-full border rounded-md py-2 px-3 text-gray-700 leading-tight 
                  focus:outline-none focus:shadow-outline"
                  defaultValue={selectedRow.yearOfBirth}
                />
              </div>

              <div className="flex justify-evenly items-center w-full">
                <label
                  htmlFor="que-quan"
                  className="w-[50%] text-gray-700 font-medium mb-2"
                >
                  Quê quán:
                </label>
                <input
                  {...register("hometown")}
                  type="text"
                  id="que-quan"
                  className="w-full border rounded-md py-2 px-3 text-gray-700 leading-tight 
                  focus:outline-none focus:shadow-outline"
                  defaultValue={selectedRow.hometown}
                />
              </div>

              <div className="flex justify-evenly items-center w-full">
                <label
                  htmlFor="so-can-cuoc"
                  className="w-[50%] text-gray-700 font-medium mb-2"
                >
                  Số căn cước:
                </label>
                <input
                  {...register("IdNumber")}
                  type="text"
                  id="so-can-cuoc"
                  className="w-full border rounded-md py-2 px-3 text-gray-700 leading-tight 
                  focus:outline-none focus:shadow-outline"
                  defaultValue={selectedRow.IdNumber}
                />
              </div>

              {/* <div className="flex justify-evenly items-center w-full">
                <label
                  htmlFor="vi-pham"
                  className="w-[50%] text-gray-700 font-medium mb-2"
                >
                  Vi phạm:
                </label>
                <input
                  {...register("violation")}
                  type="text"
                  id="vi-pham"
                  className="w-full border rounded-md py-2 px-3 text-gray-700 leading-tight 
                  focus:outline-none focus:shadow-outline"
                  defaultValue={selectedRow.violation}
                />
              </div> */}

              <div className="flex justify-evenly items-center w-full">
                <label
                  htmlFor="images-for-model"
                  className="w-[60%] text-gray-700 font-medium mb-2"
                >
                  Thêm dữ liệu model:
                </label>
                <div className="w-full flex justify-center items-center">
                  <Button
                    variant="contained"
                    component="label"
                    htmlFor="images-for-model"
                  >
                    <AddPhotoAlternateIcon />
                    <input
                      id="images-for-model"
                      name="image"
                      multiple
                      accept="image/*"
                      style={{ display: "none" }}
                      type="file"
                      onChange={(event) =>
                        onChangeHandleMultipleImageImport(event)
                      }
                    />
                  </Button>
                </div>
                {imagesFilesForModel && imagesFilesForModel.length > 0 && (
                  <p>Đã tải lên: {imagesFilesForModel.length} ảnh</p>
                )}
              </div>
              <div className="flex justify-evenly items-center w-full">
                <Button
                  onClick={() => {
                    setOpenEditBlacklistData(false);
                  }}
                >
                  Hủy
                </Button>

                <Button disabled={isSubmitting} type="submit">
                  Xác nhận
                </Button>
              </div>
            </form>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
};

export default Blacklist;
