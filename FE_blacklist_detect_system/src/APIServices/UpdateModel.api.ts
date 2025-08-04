import customAxios from "../utils/customAxios";

export const postData = async () => {
  try {
    const response = await customAxios.post("update_model/");
    return response;
  } catch (error) {
    console.error("UpdateModelServiceError: ", error);
  }
};
